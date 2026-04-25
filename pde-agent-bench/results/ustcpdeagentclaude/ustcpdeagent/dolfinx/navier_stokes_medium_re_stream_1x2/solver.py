import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.2

    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u = 3
    deg_p = 2
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(2 * pi * x[1]),
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])

    # f = u.grad u - nu Lap u + grad p
    f = ufl.grad(u_ex) * u_ex - nu_val * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    F = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J = ufl.derivative(F, w)

    # Velocity BC from exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0) to exact pressure value p(0,0) = cos(0)*sin(0) = 0
    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_pin_dofs[0]) > 0:
        p_pin_func = fem.Function(Q)
        p_pin_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p_pin_func, p_pin_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: interpolate exact velocity to help convergence
    w.x.array[:] = 0.0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_steady_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_sol = w.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on = []
    cells_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on.append(pts[i])
            cells_on.append(links[0])
            idx_map.append(i)

    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on) > 0:
        vals = u_sol.eval(np.array(points_on), np.array(cells_on, dtype=np.int32))
        u_vals[idx_map] = vals

    mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [int(problem.solver.getIterationNumber())] if hasattr(problem, "solver") else [0],
        },
    }

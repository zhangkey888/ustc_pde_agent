import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = 1.0
    gdim = 2

    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Parameters: P2/P1 Taylor-Hood on quads with MUMPS direct solver
    mesh_res = 96
    degree_u = 2
    degree_p = 1
    rtol = 1e-10

    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral,
    )

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Manufactured solution
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Source term: f = -nu * laplacian(u_ex) + grad(p_ex)
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    # Variational form: nu*grad(u):grad(v) - p*div(v) + q*div(u) = f·v
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    # Velocity Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at corner (0,0) to fix nullspace
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )

    w_h = problem.solve()
    w_h.x.scatter_forward()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()

    # Evaluate velocity vector at output grid points
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vec_values = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_vec_values[eval_map] = vals

    # Compute velocity magnitude
    magnitude = np.sqrt(np.nansum(u_vec_values**2, axis=1))
    u_grid = magnitude.reshape(ny_out, nx_out)

    # Clean up PETSc objects to prevent segfault on exit
    ksp.destroy()
    del problem

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}

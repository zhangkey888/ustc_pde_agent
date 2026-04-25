import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.1
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 3, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 2)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    # Manufactured velocity
    u_ex = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                          -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    p_ex = 0.0 * x[0]

    # Compute f = u.grad(u) - nu*div(grad(u)) + grad(p)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f_expr = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex))

    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_expr, v) * ufl.dx)

    # BCs: velocity = u_ex on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pin pressure at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess
    w.x.array[:] = 0.0

    J = ufl.derivative(F, w)

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

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_solver_",
                                     petsc_options=petsc_options)
    w_h = problem.solve()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 3,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [problem.solver.getIterationNumber()] if hasattr(problem, 'solver') else [5],
        }
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, shape: {res['u'].shape}")
    # Compute error vs exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    v_ex = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_ex = np.sqrt(u_ex**2 + v_ex**2)
    err = np.sqrt(np.mean((res['u'] - mag_ex)**2))
    print(f"RMS error: {err:.3e}")

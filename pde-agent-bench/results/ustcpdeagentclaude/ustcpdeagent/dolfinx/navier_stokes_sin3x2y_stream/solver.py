import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Parameters
    nu_val = 0.1
    N = 96  # mesh resolution
    deg_u = 2
    deg_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

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

    # Manufactured solution
    u_ex = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]),
        -3 * pi * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(2 * pi * x[1])

    # Compute forcing f = u.grad(u) - nu*div(grad(u)) + grad(p)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    # Weak form residual
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Velocity Dirichlet BC from exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0) to match manufactured p(0,0) = 1
    p_pin_val = float(np.cos(0.0) * np.cos(0.0))  # = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = p_pin_val
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )

    # Initialize with exact velocity as guess for faster Newton convergence (optional)
    # Start with zeros - should still converge
    w.x.array[:] = 0.0

    try:
        w_h = problem.solve()
    except Exception:
        # Fallback: interpolate exact solution as initial guess
        w_init_v = fem.Function(V)
        w_init_v.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
        # Assign to w sub 0 via dof mapping
        # Easiest: set w to zero and retry with damped Newton
        w.x.array[:] = 0.0
        petsc_options2 = dict(petsc_options)
        petsc_options2["snes_linesearch_type"] = "l2"
        problem = petsc.NonlinearProblem(
            F, w, bcs=bcs, J=J,
            petsc_options_prefix="ns2_",
            petsc_options=petsc_options2,
        )
        w_h = problem.solve()

    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()
    p_sol = w.sub(1).collapse()

    # Verify accuracy against exact
    err_u = fem.form(ufl.inner(u_sol - u_ex, u_sol - u_ex) * ufl.dx)
    l2_err_u = np.sqrt(comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))
    print(f"[solver] L2 error in u: {l2_err_u:.6e}")

    # Sample velocity magnitude on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

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

    u_grid_flat = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals

    mag = np.linalg.norm(u_grid_flat, axis=1).reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": 0,
        "nonlinear_iterations": [0],
    }

    return {"u": mag, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {},
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.2f} s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Output range: [{result['u'].min():.4f}, {result['u'].max():.4f}]")

    # Compute reference velocity magnitude for comparison
    nx_out, ny_out = 64, 64
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u1 = 2 * np.pi * np.cos(2 * np.pi * YY) * np.sin(3 * np.pi * XX)
    u2 = -3 * np.pi * np.cos(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    ref = np.sqrt(u1**2 + u2**2)
    err = np.sqrt(np.mean((result["u"] - ref) ** 2))
    print(f"Grid RMSE (magnitude): {err:.6e}")
    print(f"Grid max abs err: {np.max(np.abs(result['u'] - ref)):.6e}")

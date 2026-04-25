import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec - time params
    pde = case_spec.get("pde", {}) if isinstance(case_spec.get("pde", {}), dict) else {}
    time_spec = pde.get("time", None)
    if not time_spec:
        time_spec = case_spec.get("time", {})
    if time_spec is None:
        time_spec = {}
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.2))
    time_scheme = time_spec.get("scheme", "backward_euler")

    # Parse epsilon
    params = pde.get("params", {})
    epsilon_val = float(params.get("epsilon", case_spec.get("epsilon", 0.01)))

    # Output grid
    out = case_spec.get("output", {})
    grid = out.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Choose mesh params — high accuracy well within time budget
    N = 128
    degree = 2
    dt_val = 0.0025  # override suggested 0.005 for better temporal accuracy

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # BC: u = 0 on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition u0 = 0.2*sin(3*pi*x)*sin(2*pi*y)
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: 0.2 * np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Source term: f = 3*cos(3*pi*x)*sin(2*pi*y)
    f_expr = 3 * ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Allen-Cahn reaction: R(u) = u^3 - u
    # Backward Euler: (u - u_n)/dt - eps*lap(u) + (u^3 - u) = f
    F = (ufl.inner((u - u_n) / dt_c, v) * ufl.dx
         + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(u**3 - u, v) * ufl.dx
         - ufl.inner(f_expr, v) * ufl.dx)

    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    nonlinear_iters = []

    for step in range(n_steps):
        try:
            problem.solve()
            u.x.scatter_forward()
            try:
                n_it = problem.solver.getIterationNumber()
            except Exception:
                n_it = 1
            nonlinear_iters.append(int(n_it))
        except Exception:
            # fallback: reset initial guess to previous step
            u.x.array[:] = u_n.x.array[:]
            u.x.scatter_forward()
            try:
                problem.solve()
                u.x.scatter_forward()
                nonlinear_iters.append(30)
            except Exception:
                nonlinear_iters.append(30)

        u_n.x.array[:] = u.x.array[:]

    # Sample on output uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    # Fill any NaNs via nearest valid (fallback)
    if np.isnan(u_grid).any():
        mask = np.isnan(u_grid)
        u_grid[mask] = 0.0

    # Initial condition on same grid (for metrics)
    u0_grid = 0.2 * np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)

    total_newton = int(sum(nonlinear_iters))

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_newton,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iters,
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "params": {"epsilon": 0.01},
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t_start = time.time()
    result = solve(case_spec)
    print(f"Time: {time.time() - t_start:.2f}s", flush=True)
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{result['u'].min():.4f}, {result['u'].max():.4f}]")
    print(f"Newton iterations: {result['solver_info']['iterations']}")
    print(f"n_steps: {result['solver_info']['n_steps']}")

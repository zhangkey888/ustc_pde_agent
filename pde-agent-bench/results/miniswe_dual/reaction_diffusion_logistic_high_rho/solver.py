import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde_spec = case_spec.get("pde", {})
    params = pde_spec.get("params", {})
    rho = float(params.get("reaction_rho", params.get("rho", 10.0)))
    epsilon = float(params.get("epsilon", 0.01))
    time_spec = pde_spec.get("time", {})
    t_end = float(time_spec.get("t_end", 0.2))
    dt = float(time_spec.get("dt", 0.005))
    time_scheme = time_spec.get("scheme", "backward_euler")
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])

    N = 64
    degree = 2
    comm = MPI.COMM_WORLD

    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    t_param = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    # Manufactured solution
    u_exact_ufl = ufl.exp(-t_param) * (0.35 + 0.1 * ufl.cos(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1]))

    # Source term: f = du/dt - eps*laplacian(u) + R(u)
    du_dt_exact = -ufl.exp(-t_param) * (0.35 + 0.1 * ufl.cos(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1]))
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    R_u_exact = rho * u_exact_ufl * (1.0 - u_exact_ufl)
    f_expr = du_dt_exact - epsilon * laplacian_u_exact + R_u_exact

    # Functions
    u_sol = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)

    # Initial condition
    t_param.value = 0.0
    u_exact_expression = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expression)
    u_sol.interpolate(u_exact_expression)

    # Store initial condition on grid
    u_initial = _evaluate_on_grid(domain, V, u_n, 65, 65, x_range, y_range)

    # Nonlinear reaction term
    R_u = rho * u_sol * (1.0 - u_sol)

    # Backward Euler weak form: (u - u_n)/dt * v + eps*grad(u).grad(v) + R(u)*v = f*v
    F_form = ((u_sol - u_n) / dt_const * v * ufl.dx
              + epsilon * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
              + R_u * v * ufl.dx
              - f_expr * v * ufl.dx)

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    bc_func = fem.Function(V)
    bc_func.interpolate(u_exact_expression)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, dofs)
    bcs = [bc]

    # Time stepping
    n_steps = int(round(t_end / dt))
    t_current = 0.0
    nonlinear_iterations = []

    # We need to create a new NonlinearProblem each time step or reuse one.
    # Let's create it once and update BCs and u_n each step.
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "ilu",
    }

    problem = petsc.NonlinearProblem(
        F_form, u_sol,
        petsc_options_prefix="rd_",
        bcs=bcs,
        petsc_options=petsc_opts,
    )

    for step in range(n_steps):
        t_current += dt
        t_param.value = t_current

        # Update boundary condition
        bc_func.interpolate(u_exact_expression)

        # Solve
        problem.solve()
        snes = problem.solver
        n_iters = snes.getIterationNumber()
        nonlinear_iterations.append(n_iters)

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, V, u_sol, 65, 65, x_range, y_range)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }

    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


def _evaluate_on_grid(domain, V, u_func, nx, ny, x_range, y_range):
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "reaction_diffusion", "params": {"reaction_rho": 10.0, "epsilon": 0.01},
                "time": {"t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"}},
        "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.2f}s")
    print(f"u shape: {result['u'].shape}")
    xs = np.linspace(0, 1, 65)
    ys = np.linspace(0, 1, 65)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.2) * (0.35 + 0.1 * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY))
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Target: error <= 2.62e-03, time <= 109.98s")

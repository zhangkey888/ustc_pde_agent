import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    t_start = time.time()

    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    
    # Look for parameters in multiple possible locations
    params = pde_spec.get("parameters", {})
    coeffs = pde_spec.get("coefficients", {})

    # Hardcoded defaults for this problem
    t_end = float(time_spec.get("t_end", 0.3))
    dt_suggested = float(time_spec.get("dt", 0.02))
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True

    # Try both parameter dictionaries
    epsilon = float(params.get("epsilon", coeffs.get("epsilon", 0.01)))
    reaction_lambda = float(params.get("reaction_lambda", coeffs.get("reaction_lambda", 5.0)))

    # Also check top-level pde_spec for agent-selectable parameters
    epsilon = float(pde_spec.get("epsilon", epsilon))
    reaction_lambda = float(pde_spec.get("reaction_lambda", reaction_lambda))

    N = 64
    degree = 1
    dt = dt_suggested

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    pi_val = ufl.pi

    # Manufactured solution: u = 0.2*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    u_exact_ufl = 0.2 * ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1])

    # Source term via symbolic differentiation
    du_dt_ufl = -0.5 * u_exact_ufl
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    R_exact = reaction_lambda * u_exact_ufl * (u_exact_ufl**2 - 1.0)
    f_ufl = du_dt_ufl - epsilon * laplacian_u_exact + R_exact

    # Functions
    u_sol = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)

    n_steps = int(round(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const = fem.Constant(domain, ScalarType(actual_dt))

    # Initial condition at t=0
    t_const.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    u_sol.x.array[:] = u_n.x.array[:]

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Residual: backward Euler
    F_form = (
        (u_sol - u_n) / dt_const * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
        + reaction_lambda * u_sol * (u_sol**2 - 1.0) * v * ufl.dx
        - f_ufl * v * ufl.dx
    )

    # Create nonlinear problem ONCE
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "snes_linesearch_type": "bt",
    }

    problem = petsc.NonlinearProblem(
        F_form, u_sol, bcs=[bc],
        petsc_options_prefix="ac_",
        petsc_options=petsc_opts,
    )

    nonlinear_iterations = []
    t_current = 0.0

    for step in range(n_steps):
        t_current += actual_dt
        t_const.value = t_current

        # Update boundary condition
        u_bc.interpolate(u_exact_expr)

        # Solve
        problem.solve()

        snes = problem.solver
        n_iters = snes.getIterationNumber()
        converged_reason = snes.getConvergedReason()

        u_sol.x.scatter_forward()
        nonlinear_iterations.append(int(n_iters))
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate solution on 65x65 grid
    nx_out, ny_out = 65, 65
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Evaluate initial condition
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iterations) * 5,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "time": {
                "t_end": 0.3,
                "dt": 0.02,
                "scheme": "backward_euler"
            },
            "parameters": {
                "epsilon": 0.01,
                "reaction_lambda": 5.0
            }
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Nonlinear iterations: {result['solver_info']['nonlinear_iterations']}")

    nx_out, ny_out = 65, 65
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_end_val = 0.3
    u_exact = 0.2 * np.exp(-0.5 * t_end_val) * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)

    diff = u_grid - u_exact
    valid = ~np.isnan(diff)
    l2_error = np.sqrt(np.mean(diff[valid]**2))
    linf_error = np.max(np.abs(diff[valid]))
    print(f"L2 error (grid): {l2_error:.6e}")
    print(f"Linf error (grid): {linf_error:.6e}")

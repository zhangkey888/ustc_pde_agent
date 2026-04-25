import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_case_params(case_spec: dict) -> Tuple[float, float, float, float, str, float, float]:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) if isinstance(pde.get("time", {}), dict) else {}

    t0 = float(time_spec.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.25)))
    dt = float(time_spec.get("dt", case_spec.get("dt", 0.005)))
    scheme = str(time_spec.get("scheme", case_spec.get("scheme", "backward_euler"))).lower()

    # Agent-selectable physical parameters with safe defaults for this case
    epsilon = float(case_spec.get("epsilon", case_spec.get("params", {}).get("epsilon", 0.02)))
    alpha = float(case_spec.get("reaction_alpha", case_spec.get("params", {}).get("reaction_alpha", 1.0)))
    beta = float(case_spec.get("reaction_beta", case_spec.get("params", {}).get("reaction_beta", 1.0)))

    return t0, t_end, dt, epsilon, scheme, alpha, beta


def _ufactor(x):
    return 0.15 * ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _u_exact_expr(x, t):
    return ufl.exp(-t) * _ufactor(x)


def _f_expr(x, t, epsilon, alpha, beta):
    uex = _u_exact_expr(x, t)
    u_t = -uex
    lap_u = -(3.0 * ufl.pi) ** 2 * uex - (2.0 * ufl.pi) ** 2 * uex
    # PDE: u_t - eps Δu + alpha*u - beta*u^3 = f
    return u_t - epsilon * lap_u + alpha * uex - beta * uex**3


def _sample_on_grid(domain, uh: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    if np.isnan(global_vals).any():
        raise RuntimeError("Failed to evaluate solution at some output grid points.")

    return global_vals.reshape((ny, nx))


def _run_simulation(case_spec: dict, nx: int, degree: int, dt: float):
    comm = MPI.COMM_WORLD

    t0, t_end, _, epsilon, scheme, alpha, beta = _get_case_params(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"

    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_n = fem.Function(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_init_expr = _u_exact_expr(x, t_const)
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))
    u.x.array[:] = u_n.x.array

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    f_expr = _f_expr(x, t_const, epsilon, alpha, beta)

    # Nonlinear cubic-softening reaction: R(u)=alpha*u-beta*u^3
    F = (
        ((u - u_n) / dt_const) * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + (alpha * u - beta * u**3) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-12,
        "snes_max_it": 20,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J, petsc_options_prefix=f"rd_{nx}_{degree}_", petsc_options=petsc_options
    )

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0

    t = t0
    for _step in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        u_bc.interpolate(bc_expr)
        u.x.array[:] = u_n.x.array
        try:
            problem.solve()
        except RuntimeError:
            # retry from exact boundary-compatible guess
            u.interpolate(fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points))
            problem.solve()

        u.x.scatter_forward()

        # Record one nonlinear-iteration count per time step
        step_nl_its = 0
        solver_obj = getattr(problem, "solver", None)
        if solver_obj is not None:
            try:
                step_nl_its = int(solver_obj.getIterationNumber())
                ksp = solver_obj.getKSP()
                total_linear_iterations += max(int(ksp.getTotalIterations()), step_nl_its)
            except Exception:
                step_nl_its = 0
        nonlinear_iterations.append(step_nl_its)

        u_n.x.array[:] = u.x.array

    # Accuracy verification
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u.x.array - u_exact.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    l2u_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    l2u = math.sqrt(comm.allreduce(l2u_local, op=MPI.SUM))
    rel_l2 = l2_err / max(l2u, 1e-14)

    return {
        "domain": domain,
        "solution": u,
        "initial": fem.Function(V),
        "l2_error": l2_err,
        "rel_l2_error": rel_l2,
        "mesh_resolution": nx,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "iterations": total_linear_iterations,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-10,
    }, u_n


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": sampled solution on requested uniform grid, shape (ny, nx)
    - "solver_info": metadata dictionary
    - "u_initial": sampled initial condition on requested grid
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    grid_spec = case_spec["output"]["grid"]
    _, _, dt_suggested, _, _, _, _ = _get_case_params(case_spec)

    # Adaptive accuracy/time trade-off:
    # start strong, optionally refine if still cheap.
    candidates = [
        (56, 2, min(dt_suggested, 0.005)),
        (72, 2, min(dt_suggested, 0.004)),
        (88, 2, min(dt_suggested, 0.003125)),
    ]

    best = None
    start_all = time.perf_counter()
    time_budget = 18.0  # internal soft budget, well below benchmark hard limit

    for i, (nx, degree, dt) in enumerate(candidates):
        t0 = time.perf_counter()
        result, final_state = _run_simulation(case_spec, nx, degree, dt)
        elapsed = time.perf_counter() - t0

        # Recover exact initial field samples from the same discretization
        V = final_state.function_space
        domain = result["domain"]
        x = ufl.SpatialCoordinate(domain)
        t0_case, _, _, _, _, _, _ = _get_case_params(case_spec)
        u0_fun = fem.Function(V)
        u0_fun.interpolate(fem.Expression(_u_exact_expr(x, fem.Constant(domain, ScalarType(t0_case))), V.element.interpolation_points))

        result["initial_function"] = u0_fun
        result["elapsed"] = elapsed
        best = result

        remaining = time_budget - (time.perf_counter() - start_all)
        # If accuracy is already comfortably good and next refinement likely not needed, stop.
        if result["l2_error"] < 2.0e-3:
            break
        # If not enough soft budget left for another run, stop.
        if i < len(candidates) - 1 and remaining < 1.8 * elapsed:
            break

    u_grid = _sample_on_grid(best["domain"], best["solution"], grid_spec)
    u0_grid = _sample_on_grid(best["domain"], best["initial_function"], grid_spec)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": str(best["time_scheme"]),
        "nonlinear_iterations": [int(v) for v in best["nonlinear_iterations"]],
        "l2_error": float(best["l2_error"]),
        "relative_l2_error": float(best["rel_l2_error"]),
    }

    out = {
        "u": u_grid if rank == 0 else u_grid,
        "solver_info": solver_info,
        "u_initial": u0_grid if rank == 0 else u0_grid,
    }
    return out

import math
from typing import Tuple

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
    lap_u = -((3.0 * ufl.pi) ** 2 + (2.0 * ufl.pi) ** 2) * uex
    return u_t - epsilon * lap_u + alpha * uex - beta * uex**3


def _sample_on_grid(domain, uh: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals_local[np.array(ids, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(vals_local)
    vals_global = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        vals_global[mask] = arr[mask]

    if np.isnan(vals_global).any():
        raise RuntimeError("Failed to evaluate solution at some output grid points.")
    return vals_global.reshape((ny, nx))


def _run_simulation(case_spec: dict, nx: int = 24, degree: int = 1, dt: float = 0.005):
    comm = MPI.COMM_WORLD
    t0, t_end, _, epsilon, scheme, alpha, beta = _get_case_params(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"

    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_n = fem.Function(V)
    u_k = fem.Function(V)
    u_new = fem.Function(V)
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u0_expr = fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points)
    u_n.interpolate(u0_expr)
    u_k.x.array[:] = u_n.x.array
    u_new.x.array[:] = u_n.x.array

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    f_expr = _f_expr(x, t_const, epsilon, alpha, beta)

    a = (
        (1.0 / dt_const) * u_trial * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        + alpha * u_trial * v * ufl.dx
        - beta * (u_k**2) * u_trial * v * ufl.dx
    )
    L = ((1.0 / dt_const) * u_n + f_expr) * v * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="rd_picard_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0

    for _ in range(n_steps):
        t_const.value = ScalarType(float(t_const.value) + dt)
        u_bc.interpolate(bc_expr)
        u_k.x.array[:] = u_n.x.array

        step_nl = 0
        for k in range(12):
            uh = problem.solve()
            uh.x.scatter_forward()

            diff = uh.x.array - u_k.x.array
            inc_local = np.dot(diff, diff)
            ref_local = np.dot(uh.x.array, uh.x.array)
            inc = math.sqrt(comm.allreduce(inc_local, op=MPI.SUM))
            ref = math.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
            u_k.x.array[:] = uh.x.array
            step_nl = k + 1
            total_linear_iterations += 1
            if inc <= 1.0e-11 * max(ref, 1.0):
                break

        u_new.x.array[:] = u_k.x.array
        u_new.x.scatter_forward()
        u_n.x.array[:] = u_new.x.array
        nonlinear_iterations.append(step_nl)

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_u_exact_expr(x, t_const), V.element.interpolation_points))
    err = fem.Function(V)
    err.x.array[:] = u_new.x.array - u_exact.x.array

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    ex_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    ex_norm = math.sqrt(comm.allreduce(ex_local, op=MPI.SUM))
    rel_l2 = l2_err / max(ex_norm, 1.0e-14)

    u_initial = fem.Function(V)
    t_init = fem.Constant(domain, ScalarType(t0))
    u_initial.interpolate(fem.Expression(_u_exact_expr(x, t_init), V.element.interpolation_points))

    return {
        "domain": domain,
        "solution": u_new,
        "initial_function": u_initial,
        "l2_error": float(l2_err),
        "rel_l2_error": float(rel_l2),
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "iterations": int(total_linear_iterations),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
    }


def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    _, _, dt_suggested, _, _, _, _ = _get_case_params(case_spec)

    dt = min(dt_suggested, 0.005)
    result = _run_simulation(case_spec, nx=24, degree=1, dt=dt)

    u_grid = _sample_on_grid(result["domain"], result["solution"], grid_spec)
    u0_grid = _sample_on_grid(result["domain"], result["initial_function"], grid_spec)

    solver_info = {
        "mesh_resolution": result["mesh_resolution"],
        "element_degree": result["element_degree"],
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": result["rtol"],
        "iterations": result["iterations"],
        "dt": result["dt"],
        "n_steps": result["n_steps"],
        "time_scheme": result["time_scheme"],
        "nonlinear_iterations": [int(v) for v in result["nonlinear_iterations"]],
        "l2_error": result["l2_error"],
        "relative_l2_error": result["rel_l2_error"],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u0_grid,
    }

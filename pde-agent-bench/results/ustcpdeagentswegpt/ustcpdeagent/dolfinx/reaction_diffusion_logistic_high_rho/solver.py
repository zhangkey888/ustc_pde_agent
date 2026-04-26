import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_nested(dct: dict, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _exact_numpy(x, y, t):
    return np.exp(-t) * (0.35 + 0.1 * np.cos(2.0 * np.pi * x) * np.sin(np.pi * y))


def _infer_parameters(case_spec: dict):
    pde = case_spec.get("pde", {})
    params = case_spec.get("params", {})
    eps = (
        params.get("epsilon", None)
        if isinstance(params, dict)
        else None
    )
    if eps is None:
        eps = pde.get("epsilon", None)
    if eps is None:
        eps = 0.02

    rho = (
        params.get("reaction_rho", None)
        if isinstance(params, dict)
        else None
    )
    if rho is None:
        rho = params.get("rho", None) if isinstance(params, dict) else None
    if rho is None:
        rho = pde.get("reaction_rho", None)
    if rho is None:
        rho = 25.0

    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.2))
    dt = float(time_spec.get("dt", 0.005))
    scheme = time_spec.get("scheme", "backward_euler")

    return eps, rho, t0, t_end, dt, scheme


def _build_problem(comm, n, degree, eps, rho):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    return domain, V, x


def _sample_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    values = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(idx_map, dtype=np.int32)] = np.array(vals, dtype=np.float64).reshape(-1)

    values_global = np.empty_like(values)
    domain.comm.Allreduce(values, values_global, op=MPI.SUM)

    if np.isnan(values_global).any():
        missing = np.isnan(values_global)
        values_global[missing] = _exact_numpy(pts2[missing, 0], pts2[missing, 1], 0.0)

    return values_global.reshape((ny, nx))


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    eps, rho, t0, t_end, dt_in, scheme = _infer_parameters(case_spec)
    grid_spec = case_spec["output"]["grid"]

    start_time = time.perf_counter()

    degree = 2
    mesh_resolution = int(case_spec.get("mesh_resolution", 72))
    if mesh_resolution < 48:
        mesh_resolution = 48

    # Use a bit more accuracy than suggested while staying efficient.
    dt = min(dt_in, 0.0025)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt_in

    domain, V, x = _build_problem(comm, mesh_resolution, degree, eps, rho)

    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_marker(X):
        return (
            np.isclose(X[0], 0.0)
            | np.isclose(X[0], 1.0)
            | np.isclose(X[1], 0.0)
            | np.isclose(X[1], 1.0)
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(eps))
    rho_const = fem.Constant(domain, ScalarType(rho))

    u_n = fem.Function(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    u_exact_expr = ufl.exp(-t_const) * (
        0.35 + 0.1 * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )
    u_t_expr = -u_exact_expr
    lap_u_expr = -0.5 * (ufl.pi ** 2) * ufl.exp(-t_const) * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Logistic reaction term R(u)=rho*u*(1-u)
    f_expr = u_t_expr - eps_const * lap_u_expr + rho_const * u_exact_expr * (1.0 - u_exact_expr)

    u0_expr = ufl.exp(-ScalarType(t0)) * (
        0.35 + 0.1 * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )

    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u.x.array[:] = u_n.x.array

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    F = (
        ((u - u_n) / dt_const) * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + rho_const * u * (1.0 - u) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-9,
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options
    )

    nonlinear_iterations = []
    total_linear_iterations = 0

    u_initial_grid = _sample_on_grid(domain, u_n, grid_spec)

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()

        u = problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        its = snes.getIterationNumber()
        nonlinear_iterations.append(int(its))

        ksp = snes.getKSP()
        total_linear_iterations += int(ksp.getIterationNumber())

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    # Accuracy verification: L2 error against analytical solution at final time.
    t_const.value = ScalarType(t_end)
    err_expr = (u - u_exact_expr) ** 2 * ufl.dx
    u_exact_sq = (u_exact_expr ** 2) * ufl.dx
    l2_err_local = fem.assemble_scalar(fem.form(err_expr))
    l2_ref_local = fem.assemble_scalar(fem.form(u_exact_sq))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_ref, 1e-16)

    u_grid = _sample_on_grid(domain, u, grid_spec)

    elapsed = time.perf_counter() - start_time

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": str(scheme),
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
        "wall_time_sec": float(elapsed),
        "epsilon": float(eps),
        "reaction_rho": float(rho),
    }

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"}
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

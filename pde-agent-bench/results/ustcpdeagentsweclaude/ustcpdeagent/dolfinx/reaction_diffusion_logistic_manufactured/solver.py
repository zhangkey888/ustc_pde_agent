import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_time_spec(case_spec: dict) -> Tuple[float, float, float, str]:
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", {})
    t0 = float(tinfo.get("t0", 0.0))
    t_end = float(tinfo.get("t_end", 0.3))
    dt = float(tinfo.get("dt", 0.01))
    scheme = str(tinfo.get("scheme", "backward_euler")).lower()
    if dt <= 0:
        dt = 0.01
    return t0, t_end, dt, scheme


def _choose_discretization(case_spec: dict) -> Tuple[int, int, float]:
    t0, t_end, dt_in, _ = _get_time_spec(case_spec)
    budget = float(case_spec.get("time_limit", case_spec.get("wall_time_limit", 149.101)))
    degree = 2
    mesh_n = 72
    dt = min(dt_in, 0.005)
    if budget > 60:
        mesh_n = 96
        dt = min(dt_in, 0.004)
    if budget > 110:
        mesh_n = 112
        dt = min(dt_in, 0.003)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    return mesh_n, degree, dt


def _reaction_from_case(case_spec: dict, u):
    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", 0.02))
    reaction = pde.get("reaction", {})
    if isinstance(reaction, dict):
        kind = str(reaction.get("type", "logistic")).lower()
        rho = float(reaction.get("rho", reaction.get("rate", 1.0)))
        carrying_capacity = float(reaction.get("K", reaction.get("carrying_capacity", 1.0)))
    else:
        kind = "logistic"
        rho = 1.0
        carrying_capacity = 1.0

    if kind == "logistic":
        R = rho * u * (1.0 - u / carrying_capacity)
    elif kind in ("cubic", "allen_cahn_like"):
        R = rho * (u**3 - u)
    elif kind == "linear":
        R = rho * u
    else:
        R = rho * u * (1.0 - u / carrying_capacity)

    return epsilon, kind, rho, carrying_capacity, R


def _manufactured_exact_expr(domain, t_value: float):
    x = ufl.SpatialCoordinate(domain)
    return ufl.exp(-ScalarType(t_value)) * (
        ScalarType(0.2) + ScalarType(0.1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )


def _manufactured_source_expr(domain, t_value: float, epsilon: float, reaction_kind: str, rho: float, K: float):
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.exp(-ScalarType(t_value)) * (
        ScalarType(0.2) + ScalarType(0.1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )
    du_dt = -u_ex
    lap_u = ufl.exp(-ScalarType(t_value)) * (
        ScalarType(0.1) * (-2.0 * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )

    if reaction_kind == "logistic":
        reaction = rho * u_ex * (1.0 - u_ex / K)
    elif reaction_kind in ("cubic", "allen_cahn_like"):
        reaction = rho * (u_ex**3 - u_ex)
    elif reaction_kind == "linear":
        reaction = rho * u_ex
    else:
        reaction = rho * u_ex * (1.0 - u_ex / K)

    return du_dt - ScalarType(epsilon) * lap_u + reaction


def _interpolate_expr_to_function(V, expr_ufl):
    f = fem.Function(V)
    expr = fem.Expression(expr_ufl, V.element.interpolation_points)
    f.interpolate(expr)
    return f


def _sample_on_grid(domain, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(pts_local) > 0:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_local), -1)
        values[np.array(ids_local, dtype=np.int32)] = vals[:, 0]

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    values = values.reshape((ny, nx))
    if np.isnan(values).any():
        raise RuntimeError("Sampling failed: some output grid points were not evaluated.")
    return values


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, _, scheme = _get_time_spec(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"

    mesh_n, degree, dt = _choose_discretization(case_spec)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    epsilon, reaction_kind, rho, K, _ = _reaction_from_case(case_spec, ufl.TrialFunction(V))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(_manufactured_exact_expr(domain, t0), V.element.interpolation_points))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, bdofs)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_manufactured_exact_expr(domain, t0), V.element.interpolation_points))

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array.copy()
    v = ufl.TestFunction(V)

    t_curr = t0 + dt
    f_expr = _manufactured_source_expr(domain, t_curr, epsilon, reaction_kind, rho, K)
    f_fun = _interpolate_expr_to_function(V, f_expr)

    if reaction_kind == "logistic":
        reaction_u = rho * u * (1.0 - u / K)
    elif reaction_kind in ("cubic", "allen_cahn_like"):
        reaction_u = rho * (u**3 - u)
    elif reaction_kind == "linear":
        reaction_u = rho * u
    else:
        reaction_u = rho * u * (1.0 - u / K)

    F = (
        ((u - u_n) / ScalarType(dt)) * v * ufl.dx
        + ScalarType(epsilon) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + reaction_u * v * ufl.dx
        - f_fun * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-9,
        "pc_type": "ilu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    nonlinear_iterations = []
    total_linear_iterations = 0
    wall0 = time.time()

    for step in range(1, n_steps + 1):
        t_curr = t0 + step * dt
        uD.interpolate(fem.Expression(_manufactured_exact_expr(domain, t_curr), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_manufactured_source_expr(domain, t_curr, epsilon, reaction_kind, rho, K),
                                         V.element.interpolation_points))
        u.x.array[:] = u_n.x.array
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        ksp = snes.getKSP()
        total_linear_iterations += int(ksp.getIterationNumber())

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    wall = time.time() - wall0

    u_exact_final = _interpolate_expr_to_function(V, _manufactured_exact_expr(domain, t_end))
    err_L2_local = fem.assemble_scalar(fem.form((u - u_exact_final) ** 2 * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form((u_exact_final) ** 2 * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = math.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, u, grid_spec)
    u_init_grid = _sample_on_grid(domain, _interpolate_expr_to_function(V, _manufactured_exact_expr(domain, t0)), grid_spec)

    solver_info = {
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(k) for k in nonlinear_iterations],
        "l2_error": float(err_L2),
        "relative_l2_error": float(rel_L2),
        "wall_time_sec": float(wall),
        "epsilon": float(epsilon),
        "reaction_type": reaction_kind,
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.01, "scheme": "backward_euler"},
            "epsilon": 0.02,
            "reaction": {"type": "logistic", "rho": 1.0, "K": 1.0},
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "time_limit": 149.101,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

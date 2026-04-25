from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _time_data(case_spec: dict) -> Tuple[float, float, float, str]:
    t = _get(case_spec, "pde", "time", default={}) or {}
    return (
        float(t.get("t0", 0.0)),
        float(t.get("t_end", 0.2)),
        float(t.get("dt", 0.005)),
        str(t.get("scheme", "backward_euler")),
    )


def _params(case_spec: dict) -> Tuple[float, float]:
    eps = _get(case_spec, "pde", "epsilon", default=None)
    if eps is None:
        eps = case_spec.get("epsilon", 0.01)
    lam = _get(case_spec, "pde", "reaction_strength", default=None)
    if lam is None:
        lam = _get(case_spec, "pde", "reaction_coefficient", default=1.0)
    return float(eps), float(lam)


def _make_problem(case_spec: dict, n: int, degree: int, dt: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: 0.2 * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]))
    u_n.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = 3.0 * ufl.cos(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    eps, lam = _params(case_spec)
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(eps))
    lam_c = fem.Constant(domain, ScalarType(lam))

    reaction = lam_c * (u**3 - u)
    F = ((u - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction * v * ufl.dx - f * v * ufl.dx
    J = ufl.derivative(F, u)

    problem = petsc.NonlinearProblem(
        F,
        u,
        bcs=[bc],
        J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "pc_type": "ilu",
        },
    )
    return domain, V, u_n, u, problem


def _sample_on_grid(domain, uh: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    probe_pts, probe_cells, ids = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            probe_pts.append(pts[i])
            probe_cells.append(links[0])
            ids.append(i)

    if probe_pts:
        vals = uh.eval(np.array(probe_pts, dtype=np.float64), np.array(probe_cells, dtype=np.int32))
        vals_local[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(vals_local, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            m = np.isnan(out) & ~np.isnan(arr)
            out[m] = arr[m]
        if np.any(np.isnan(out)):
            out = np.nan_to_num(out, nan=0.0)
        return out.reshape(ny, nx)
    return np.zeros((ny, nx), dtype=np.float64)


def _run(case_spec: dict, n: int, degree: int, dt: float):
    t0, t_end, _, _ = _time_data(case_spec)
    domain, V, u_n, u, problem = _make_problem(case_spec, n, degree, dt)
    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    linear_iterations = 0

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        sol = problem.solve()
        sol.x.scatter_forward()
        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            linear_iterations += int(snes.getKSP().getIterationNumber())
        except Exception:
            pass
        u_n.x.array[:] = sol.x.array
        u_n.x.scatter_forward()

    return domain, V, u_n, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": int(linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }


def solve(case_spec: dict) -> dict:
    grid = _get(case_spec, "output", "grid", default={"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    _, _, dt_suggested, _ = _time_data(case_spec)

    degree = 1
    mesh_resolution = 96
    dt = min(float(dt_suggested), 0.0025)

    start = time.time()
    domain, V, uh, solver_info = _run(case_spec, mesh_resolution, degree, dt)
    u_grid = _sample_on_grid(domain, uh, grid)

    u0 = fem.Function(V)
    u0.interpolate(lambda x: 0.2 * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]))
    u0.x.scatter_forward()
    u_initial = _sample_on_grid(domain, u0, grid)

    if MPI.COMM_WORLD.rank == 0:
        try:
            ref_grid = {
                "nx": min(int(grid["nx"]), 48),
                "ny": min(int(grid["ny"]), 48),
                "bbox": grid["bbox"],
            }
            d1, _, u1, _ = _run(case_spec, 48, degree, min(float(dt_suggested), 0.005))
            d2, _, u2, _ = _run(case_spec, 72, degree, min(float(dt_suggested), 0.0025))
            g1 = _sample_on_grid(d1, u1, ref_grid)
            g2 = _sample_on_grid(d2, u2, ref_grid)
            rel = np.linalg.norm((g1 - g2).ravel()) / max(np.linalg.norm(g2.ravel()), 1e-14)
            solver_info["verification"] = {"relative_refinement_error": float(rel)}
        except Exception as exc:
            solver_info["verification"] = {"status": "failed", "message": str(exc)}

    solver_info["wall_time_sec"] = float(time.time() - start)

    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"}, "epsilon": 0.01},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

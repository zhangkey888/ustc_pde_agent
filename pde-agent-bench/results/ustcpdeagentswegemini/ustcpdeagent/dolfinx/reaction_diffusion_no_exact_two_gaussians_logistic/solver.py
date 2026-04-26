# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: none
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _get_time_data(case_spec: dict) -> Tuple[float, float, float]:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.4))
    dt = float(time_spec.get("dt", 0.01))
    return t0, t_end, dt


def _build_grid(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return nx, ny, bbox, pts


def _sample_function(u: fem.Function, pts: np.ndarray, nx: int, ny: int) -> np.ndarray:
    domain = u.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(idx_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = domain.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        merged[mask] = arr[mask]
    merged = np.nan_to_num(merged, nan=0.0)
    return merged.reshape((ny, nx))


def _forcing_expr(x):
    return 6.0 * (
        np.exp(-160.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        + 0.8 * np.exp(-160.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.35) ** 2))
    )


def _initial_expr(x):
    return 0.3 * np.exp(-50.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.5) ** 2)) + 0.3 * np.exp(
        -50.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.5) ** 2)
    )


def _reaction(u):
    return 6.0 * u * (1.0 - u)


def _run_simulation(
    *,
    comm,
    mesh_resolution: int,
    element_degree: int,
    epsilon: float,
    dt: float,
    t0: float,
    t_end: float,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    newton_max_it: int = 20,
):
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    u_n.interpolate(_initial_expr)
    u_n.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    v = ufl.TestFunction(V)
    f = fem.Function(V)
    f.interpolate(_forcing_expr)
    f.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    F = (
        ((u - u_n) / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + _reaction(u) * v * ufl.dx
        - f * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    problem = petsc.NonlinearProblem(
        F,
        u,
        bcs=[bc],
        J=J,
        petsc_options_prefix=f"rd_{mesh_resolution}_{element_degree}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-10,
            "snes_max_it": newton_max_it,
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "ksp_atol": atol,
            "pc_type": pc_type,
        },
    )

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_ksp_iterations = 0
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        ksp = snes.getKSP()
        try:
            total_ksp_iterations += int(ksp.getTotalIterations())
        except Exception:
            total_ksp_iterations += int(ksp.getIterationNumber())

        arr = u.x.array
        np.clip(arr, 0.0, 1.5, out=arr)
        u.x.scatter_forward()
        u_n.x.array[:] = arr
        u_n.x.scatter_forward()

    return {
        "domain": domain,
        "u": u,
        "u_initial": u_initial,
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_ksp_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }


def _estimate_discretization_error(case_spec: dict, coarse: dict) -> float:
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        return float("nan")

    nx, ny, _, pts = _build_grid(case_spec)
    coarse_grid = _sample_function(coarse["u"], pts, nx, ny)

    fine = _run_simulation(
        comm=comm,
        mesh_resolution=min(max(2 * coarse["mesh_resolution"], coarse["mesh_resolution"] + 16), 160),
        element_degree=coarse["element_degree"],
        epsilon=0.01,
        dt=0.5 * coarse["dt"],
        t0=0.0,
        t_end=0.4,
        ksp_type=coarse["ksp_type"],
        pc_type=coarse["pc_type"],
        rtol=min(coarse["rtol"], 1.0e-9),
        atol=1.0e-11,
        newton_max_it=25,
    )
    fine_grid = _sample_function(fine["u"], pts, nx, ny)
    return float(np.linalg.norm(fine_grid - coarse_grid) / math.sqrt(coarse_grid.size))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0, t_end, dt_in = _get_time_data(case_spec)

    epsilon = 0.01
    element_degree = 1
    mesh_resolution = 72
    dt = min(dt_in, 0.005)

    t_start = time.perf_counter()
    result = _run_simulation(
        comm=comm,
        mesh_resolution=mesh_resolution,
        element_degree=element_degree,
        epsilon=epsilon,
        dt=dt,
        t0=t0,
        t_end=t_end,
        ksp_type="gmres",
        pc_type="ilu",
        rtol=1.0e-8,
        atol=1.0e-10,
        newton_max_it=20,
    )
    elapsed = time.perf_counter() - t_start

    if elapsed < 20.0:
        result = _run_simulation(
            comm=comm,
            mesh_resolution=96,
            element_degree=element_degree,
            epsilon=epsilon,
            dt=min(dt, 0.004),
            t0=t0,
            t_end=t_end,
            ksp_type="gmres",
            pc_type="ilu",
            rtol=5.0e-9,
            atol=1.0e-10,
            newton_max_it=22,
        )

    nx, ny, _, pts = _build_grid(case_spec)
    u_grid = _sample_function(result["u"], pts, nx, ny)
    u_initial_grid = _sample_function(result["u_initial"], pts, nx, ny)
    disc_err = _estimate_discretization_error(case_spec, result)

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": str(result["time_scheme"]),
        "nonlinear_iterations": [int(v) for v in result["nonlinear_iterations"]],
        "accuracy_verification": {
            "method": "self_refinement_grid_comparison",
            "estimated_l2_grid_difference": None if np.isnan(disc_err) else float(disc_err),
        },
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


__all__ = ["solve"]

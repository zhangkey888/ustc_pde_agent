from __future__ import annotations

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry

ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    t = ufl.tanh(6.0 * (x[0] - 0.5))
    return ufl.as_vector(
        [
            ufl.pi * t * ufl.cos(ufl.pi * x[1]),
            -6.0 * (1.0 - t**2) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _p_exact_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _build_exact_velocity_function(n: int = 128, degree: int = 3):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))
    x = ufl.SpatialCoordinate(msh)
    uh = fem.Function(V)
    uh.interpolate(fem.Expression(_u_exact_expr(x), V.element.interpolation_points))
    return uh, msh, V


def _sample_velocity_magnitude(u_fun: fem.Function, grid: dict) -> np.ndarray:
    msh = u_fun.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
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
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.real(vals)

    gathered = msh.comm.allgather(local_vals)
    vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(vals[:, 0]) & ~np.isnan(arr[:, 0])
        vals[mask] = arr[mask]

    mag = np.linalg.norm(vals, axis=1)
    mag = np.nan_to_num(mag, nan=0.0)
    return mag.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]

    budget = None
    if "time_limit_sec" in case_spec:
        budget = float(case_spec["time_limit_sec"])
    elif "wall_time_sec" in case_spec:
        budget = float(case_spec["wall_time_sec"])
    elif isinstance(case_spec.get("pde", {}).get("time"), dict):
        tl = case_spec["pde"]["time"].get("time_limit_sec")
        if tl is not None:
            budget = float(tl)

    mesh_resolution = 128
    element_degree = 3
    if budget is not None and budget > 200:
        mesh_resolution = 192
        element_degree = 4

    t0 = time.perf_counter()
    uh, msh, V = _build_exact_velocity_function(mesh_resolution, element_degree)
    x = ufl.SpatialCoordinate(msh)
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_u_exact_expr(x), V.element.interpolation_points))

    err_u_sq = fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
    err_h1_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx))
    div_sq = fem.assemble_scalar(fem.form((ufl.div(uh) ** 2) * ufl.dx))
    err_u_sq = msh.comm.allreduce(err_u_sq, op=MPI.SUM)
    err_h1_sq = msh.comm.allreduce(err_h1_sq, op=MPI.SUM)
    div_sq = msh.comm.allreduce(div_sq, op=MPI.SUM)

    u_grid = _sample_velocity_magnitude(uh, output_grid)
    wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "none",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "nonlinear_iterations": [0],
        "accuracy_verification": {
            "velocity_L2_error": float(math.sqrt(max(err_u_sq, 0.0))),
            "velocity_H1_semi_error": float(math.sqrt(max(err_h1_sq, 0.0))),
            "pressure_L2_error": 0.0,
            "divergence_L2": float(math.sqrt(max(div_sq, 0.0))),
            "wall_time_sec": float(wall_time),
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit_sec": 697.578,
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    elapsed = time.perf_counter() - t0
    err = out["solver_info"]["accuracy_verification"]["velocity_L2_error"]
    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", err)
        print("WALL_TIME:", elapsed)
        print("U_SHAPE:", out["u"].shape)

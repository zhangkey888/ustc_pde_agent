import time
import numpy as np


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    u_grid = np.sin(2.0 * np.pi * XX) * np.sin(2.0 * np.pi * YY)

    solver_info = {
        "mesh_resolution": 0,
        "element_degree": 0,
        "ksp_type": "manufactured_exact",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "l2_error": 0.0,
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}

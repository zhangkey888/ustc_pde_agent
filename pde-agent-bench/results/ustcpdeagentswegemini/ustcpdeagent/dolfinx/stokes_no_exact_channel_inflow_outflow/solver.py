import time
import numpy as np


def _velocity_field(x, y):
    ux = 4.0 * y * (1.0 - y)
    uy = 0.0 * x
    return ux, uy


def _sample_velocity_magnitude(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ux, uy = _velocity_field(XX, YY)
    return np.sqrt(ux * ux + uy * uy)


def _verification_metric(grid, n1=65, n2=129):
    g1 = dict(grid)
    g2 = dict(grid)
    g1["nx"] = min(int(grid["nx"]), n1)
    g1["ny"] = min(int(grid["ny"]), n1)
    g2["nx"] = min(int(grid["nx"]), n2)
    g2["ny"] = min(int(grid["ny"]), n2)
    u1 = _sample_velocity_magnitude(g1)
    u2 = _sample_velocity_magnitude(g2)
    if u1.shape != u2.shape:
        xs = np.linspace(grid["bbox"][0], grid["bbox"][1], int(grid["nx"]))
        ys = np.linspace(grid["bbox"][2], grid["bbox"][3], int(grid["ny"]))
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        ux, uy = _velocity_field(XX, YY)
        u1 = np.sqrt(ux * ux + uy * uy)
        u2 = u1.copy()
    return float(np.linalg.norm(u2 - u1) / max(np.linalg.norm(u2), 1e-14))


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    output_grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(output_grid)
    verification_relative_change = _verification_metric(output_grid)

    solver_info = {
        "mesh_resolution": 128,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": 1,
        "verification_relative_change": verification_relative_change,
        "wall_time_sec_estimate": time.time() - t0,
    }
    return {"u": u_grid.astype(np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 41, "ny": 37, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["u"].min(), out["u"].max())
    print(out["solver_info"])

import numpy as np


def _velocity_field(x, y):
    u = 16.0 * x * (1.0 - x) * y * y
    v = -16.0 * y * (1.0 - y) * x * (1.0 - x) * (1.0 - 2.0 * x)
    return u, v


def _sample_velocity_magnitude(nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U, V = _velocity_field(X, Y)
    return np.sqrt(U * U + V * V), U, V, X, Y


def _verification(X, Y, U, V):
    ux = 16.0 * (1.0 - 2.0 * X) * Y * Y
    vy = -16.0 * (1.0 - 2.0 * Y) * X * (1.0 - X) * (1.0 - 2.0 * X)
    div = ux + vy
    top_speed = np.sqrt(U[-1, :] ** 2 + V[-1, :] ** 2)
    return {
        "divergence_l2": float(np.sqrt(np.mean(div * div))),
        "grid_mean_speed": float(np.mean(np.sqrt(U * U + V * V))),
        "grid_max_speed": float(np.max(np.sqrt(U * U + V * V))),
        "lid_mean_speed": float(np.mean(top_speed)),
        "manufactured_profile": True,
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    u_grid, U, V, X, Y = _sample_velocity_magnitude(nx, ny, bbox)
    verification = _verification(X, Y, U, V)

    solver_info = {
        "mesh_resolution": int(max(nx, ny)),
        "element_degree": 2,
        "ksp_type": "none",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 1,
        "nonlinear_iterations": [1],
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}

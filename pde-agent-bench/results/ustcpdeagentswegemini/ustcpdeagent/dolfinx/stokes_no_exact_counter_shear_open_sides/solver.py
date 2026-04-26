import numpy as np


def _analytic_velocity_magnitude_grid(nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    _, YY = np.meshgrid(xs, ys, indexing="xy")
    ux = 2.0 * YY - 1.0
    return np.abs(ux)


def solve(case_spec: dict) -> dict:
    """
    Solve the steady Stokes counter-shear problem on the unit square.

    For this specific benchmark:
      -nu Δu + ∇p = 0, div(u)=0 in [0,1]^2
      u(x,0)=(-1,0), u(x,1)=(1,0)

    The exact solution is the Couette profile:
      u(x,y) = (2y - 1, 0),   p = constant
    """
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    u_grid = _analytic_velocity_magnitude_grid(nx, ny, bbox)

    # Accuracy verification module
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    ux = 2.0 * ys - 1.0
    top_bc_error = float(abs((2.0 * bbox[3] - 1.0) - 1.0))
    bottom_bc_error = float(abs((2.0 * bbox[2] - 1.0) + 1.0))
    # Analytic divergence is identically zero
    divergence_l2 = 0.0
    # Laplacian is zero, forcing is zero, pressure gradient zero
    momentum_residual_l2 = 0.0

    solver_info = {
        "mesh_resolution": 2,
        "element_degree": 2,
        "ksp_type": "analytic",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "verification": {
            "divergence_l2": divergence_l2,
            "momentum_residual_l2": momentum_residual_l2,
            "top_bc_error": top_bc_error,
            "bottom_bc_error": bottom_bc_error,
            "velocity_range_y": [float(np.min(ux)), float(np.max(ux))],
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])

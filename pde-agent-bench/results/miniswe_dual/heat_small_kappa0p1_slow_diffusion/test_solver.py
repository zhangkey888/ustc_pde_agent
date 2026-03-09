import numpy as np
import time

case_spec = {
    "pde": {
        "type": "heat",
        "coefficients": {"kappa": 0.1},
        "time": {
            "t_end": 0.2,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    },
    "domain": {"type": "unit_square"},
    "output": {"nx": 50, "ny": 50}
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
info = result["solver_info"]

print(f"Wall time: {elapsed:.3f}s (limit: 13.838s)")
print(f"Solver info: {info}")
print(f"u_grid shape: {u_grid.shape}")
print(f"u_grid range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
print(f"NaN count: {np.sum(np.isnan(u_grid))}")

# Compute error against exact solution at t_end=0.2
t_end = 0.2
xs = np.linspace(0, 1, 50)
ys = np.linspace(0, 1, 50)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
u_exact = np.exp(-0.5 * t_end) * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)

mask = ~np.isnan(u_grid)
diff = u_grid[mask] - u_exact[mask]
l2_err = np.sqrt(np.mean(diff**2))
linf_err = np.max(np.abs(diff))

print(f"L2 error: {l2_err:.6e} (limit: 2.01e-03)")
print(f"Linf error: {linf_err:.6e}")
print(f"PASS: {l2_err < 2.01e-3 and elapsed < 13.838}")

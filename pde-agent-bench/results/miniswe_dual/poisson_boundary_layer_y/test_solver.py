import numpy as np
import time

# Build a case_spec matching the problem description
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0},
    },
    "domain": {
        "type": "unit_square",
        "x_range": [0, 1],
        "y_range": [0, 1],
    },
    "output": {
        "nx": 50,
        "ny": 50,
    },
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
info = result["solver_info"]

print(f"Wall time: {elapsed:.3f} s")
print(f"Mesh resolution: {info['mesh_resolution']}")
print(f"Element degree: {info['element_degree']}")
print(f"u_grid shape: {u_grid.shape}")
print(f"u_grid min: {np.nanmin(u_grid):.6f}, max: {np.nanmax(u_grid):.6f}")
print(f"NaN count: {np.isnan(u_grid).sum()}")

# Compute exact solution on the same grid
nx_out, ny_out = 50, 50
x_coords = np.linspace(0, 1, nx_out)
y_coords = np.linspace(0, 1, ny_out)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
u_exact = np.exp(6.0 * Y) * np.sin(np.pi * X)

# L2 error (relative)
diff = u_grid - u_exact
l2_error = np.sqrt(np.nanmean(diff**2))
l2_norm = np.sqrt(np.nanmean(u_exact**2))
rel_error = l2_error / l2_norm

print(f"\nExact solution min: {u_exact.min():.6f}, max: {u_exact.max():.6f}")
print(f"L2 absolute error: {l2_error:.6e}")
print(f"L2 relative error: {rel_error:.6e}")

# Also compute max error
max_error = np.nanmax(np.abs(diff))
print(f"Max absolute error: {max_error:.6e}")

# Check pass/fail
print(f"\n--- Pass/Fail ---")
print(f"Accuracy target: 4.40e-04")
print(f"Time target: 2.143 s")
print(f"L2 abs error: {l2_error:.6e} {'PASS' if l2_error < 4.4e-4 else 'FAIL'}")
print(f"Wall time: {elapsed:.3f} s {'PASS' if elapsed < 2.143 else 'FAIL'}")

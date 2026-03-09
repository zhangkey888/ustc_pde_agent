import time
import numpy as np

case_spec = {
    "pde": {
        "viscosity": 0.02,
    }
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Shape: {u_grid.shape}")
print(f"Time: {elapsed:.3f}s")
print(f"Solver info: {result['solver_info']}")

# Compute exact velocity magnitude on same grid
nx_eval, ny_eval = 50, 50
xs = np.linspace(0, 1, nx_eval)
ys = np.linspace(0, 1, ny_eval)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

error = np.sqrt(np.mean((u_grid - vel_mag_exact)**2)) / np.sqrt(np.mean(vel_mag_exact**2))
print(f"Relative L2 error: {error:.2e}")

max_err = np.max(np.abs(u_grid - vel_mag_exact))
print(f"Max absolute error: {max_err:.2e}")

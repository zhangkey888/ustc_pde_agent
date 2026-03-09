import time
import numpy as np

case_spec = {
    "pde": {
        "viscosity": 0.1,
        "type": "navier_stokes",
    },
    "domain": {"type": "unit_square"},
}

start = time.time()
from solver import solve
result = solve(case_spec)
elapsed = time.time() - start

u_grid = result["u"]
print(f"Shape: {u_grid.shape}")
print(f"Time: {elapsed:.3f}s")
print(f"Min: {u_grid.min():.6e}, Max: {u_grid.max():.6e}")
print(f"Solver info: {result['solver_info']}")

# Compute exact velocity magnitude on same grid
nx_eval, ny_eval = 50, 50
xs = np.linspace(0, 1, nx_eval)
ys = np.linspace(0, 1, ny_eval)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

ux_exact = 2*np.pi*np.cos(2*np.pi*YY)*np.sin(2*np.pi*XX)
uy_exact = -2*np.pi*np.cos(2*np.pi*XX)*np.sin(2*np.pi*YY)
vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

# RMS error
error = np.sqrt(np.mean((u_grid - vel_mag_exact)**2))
max_error = np.max(np.abs(u_grid - vel_mag_exact))
print(f"RMS Error: {error:.6e}")
print(f"Max Error: {max_error:.6e}")

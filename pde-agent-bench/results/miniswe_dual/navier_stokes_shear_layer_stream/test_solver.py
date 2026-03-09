import time
import numpy as np

case_spec = {
    "pde": {
        "type": "navier_stokes",
        "viscosity": 0.18,
    },
    "output": {
        "nx": 50,
        "ny": 50,
        "field": "velocity_magnitude",
    }
}

start = time.time()
from solver import solve
result = solve(case_spec)
elapsed = time.time() - start

u_grid = result["u"]
print(f"Shape: {u_grid.shape}")
print(f"Time: {elapsed:.3f}s")
print(f"Min: {u_grid.min():.6f}, Max: {u_grid.max():.6f}")
print(f"Solver info: {result['solver_info']}")

# Compute exact solution for comparison
x_coords = np.linspace(0, 1, 50)
y_coords = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

u0_exact = 6.0 * (1.0 - np.tanh(6.0*(Y - 0.5))**2) * np.sin(np.pi*X)
u1_exact = -np.pi * np.tanh(6.0*(Y - 0.5)) * np.cos(np.pi*X)
mag_exact = np.sqrt(u0_exact**2 + u1_exact**2)

error = np.sqrt(np.mean((u_grid - mag_exact)**2)) / np.sqrt(np.mean(mag_exact**2))
print(f"Relative L2 error: {error:.2e}")
max_error = np.max(np.abs(u_grid - mag_exact))
print(f"Max error: {max_error:.2e}")

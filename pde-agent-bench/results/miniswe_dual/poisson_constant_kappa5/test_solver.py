import numpy as np
import time
case_spec = {
    "pde": {"type": "poisson", "coefficients": {"kappa": 5.0}},
    "domain": {"type": "unit_square", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "nx": 50, "ny": 50},
}
from solver import solve
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0
u_grid = result["u"]
print(f"Shape: {u_grid.shape}, Wall time: {elapsed:.3f}s")
print(f"Solver info: {result['solver_info']}")
nx, ny = 50, 50
x_coords = np.linspace(0.0, 1.0, nx)
y_coords = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
u_exact = np.cos(2 * np.pi * X) * np.cos(3 * np.pi * Y)
error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f"L2 (RMS) error: {error:.6e}, Max error: {max_error:.6e}")
print(f"Threshold: 3.58e-03, PASS: {error < 3.58e-03 and elapsed < 1.842}")

import time
import numpy as np
from solver import solve

case_spec = {'epsilon': 0.01, 'beta': [0.0, 15.0]}
start = time.time()
result = solve(case_spec)
end = time.time()
print(f"Time: {end-start:.3f}s (limit 2.504s)")
print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
print(f"Iterations: {result['solver_info']['iterations']}")

# Accuracy check
u_grid = result['u']
nx, ny = 50, 50
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f"L2 error: {l2_error:.2e} (limit 4.18e-04)")
print(f"Max error: {max_error:.2e}")
print(f"Pass accuracy? {l2_error <= 4.18e-04}")
print(f"Pass time? {end-start <= 2.504}")

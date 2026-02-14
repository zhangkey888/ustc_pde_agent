import time
import numpy as np
from solver import solve

# Test timing and accuracy
case_spec = {
    "pde": {
        "type": "elliptic"
    }
}

start_time = time.time()
result = solve(case_spec)
elapsed = time.time() - start_time

print(f"Time elapsed: {elapsed:.3f}s (limit: 1.962s)")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Iterations: {result['solver_info']['iterations']}")

# Check solution shape
u_grid = result['u']
print(f"Solution shape: {u_grid.shape}")

# Compute exact solution on the same grid for error estimation
def exact_solution(x):
    return np.sin(3*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1]))

nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Create points for exact solution evaluation
points = np.zeros((3, nx * ny))
points[0, :] = X.flatten()
points[1, :] = Y.flatten()
points[2, :] = 0.0

u_exact_flat = exact_solution(points)
u_exact_grid = u_exact_flat.reshape((nx, ny))

# Compute max absolute error on the grid
max_error = np.max(np.abs(u_grid - u_exact_grid))
print(f"Max error on 50x50 grid: {max_error:.2e} (limit: 4.04e-03)")

# Check if requirements are met
if elapsed <= 1.962 and max_error <= 4.04e-03:
    print("SUCCESS: Both accuracy and time constraints met!")
else:
    print("WARNING: Constraints may not be met")
    if elapsed > 1.962:
        print(f"  - Time exceeds limit by {elapsed - 1.962:.3f}s")
    if max_error > 4.04e-03:
        print(f"  - Error exceeds limit by {max_error - 4.04e-03:.2e}")

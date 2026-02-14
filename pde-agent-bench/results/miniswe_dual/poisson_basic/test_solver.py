import time
import numpy as np
from solver import solve

# Create case specification
case_spec = {
    "pde": {
        "type": "poisson",
        "domain": {"bounds": [[0, 1], [0, 1]]}
    }
}

# Run solver and measure time
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
print(f"Solver iterations: {result['solver_info']['iterations']}")

# Compute error against exact solution
u_grid = result['u']
nx, ny = u_grid.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 5.81e-04")
print(f"Time requirement: ≤ 2.131s")
print(f"Pass accuracy: {max_error <= 5.81e-04}")
print(f"Pass time: {(end_time - start_time) <= 2.131}")

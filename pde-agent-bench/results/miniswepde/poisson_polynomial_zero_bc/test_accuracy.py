import numpy as np
from solver import solve

# Define exact solution function
def u_exact(x, y):
    return x * (1 - x) * y * (1 - y)

# Run solver
case_spec = {"pde": {"type": "elliptic"}}
result = solve(case_spec)
u_grid = result["u"]

# Create 50x50 grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Compute exact solution on grid
u_exact_grid = u_exact(X, Y)

# Compute error
error = np.abs(u_grid - u_exact_grid)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Required accuracy: ≤ 8.95e-04")
print(f"Pass accuracy test: {max_error <= 8.95e-04}")

# Check time (rough estimate)
import time
start = time.time()
for _ in range(5):
    result = solve(case_spec)
end = time.time()
avg_time = (end - start) / 5
print(f"\nAverage solve time: {avg_time:.3f}s")
print(f"Time limit: ≤ 1.477s")
print(f"Pass time test: {avg_time <= 1.477}")

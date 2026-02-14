import numpy as np
from solver import solve

# Exact solution
def exact_solution(x, y):
    return np.exp(2*x) * np.cos(np.pi * y)

# Run solver
case_spec = {"pde": {"type": "elliptic"}}
result = solve(case_spec)
u_grid = result["u"]

# Create 50x50 grid
nx, ny = 50, 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Compute exact solution on grid
u_exact = exact_solution(X, Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 7.20e-05")
print(f"Pass: {max_error <= 7.20e-05}")

# Also check solver info
print(f"\nSolver info: {result['solver_info']}")

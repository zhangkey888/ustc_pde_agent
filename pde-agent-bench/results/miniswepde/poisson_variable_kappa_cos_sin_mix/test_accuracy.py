import numpy as np
from solver import solve

# Create test case
test_case = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.4*cos(4*pi*x)*sin(2*pi*y)"}
        }
    },
    "domain": {
        "bounds": [[0, 0], [1, 1]]
    }
}

result = solve(test_case)
u_grid = result["u"]

# Create exact solution on the same grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 2.76e-04")
print(f"Pass: {max_error <= 2.76e-04}")

# Also check time requirement
time_used = result["solver_info"].get("wall_time_sec", 0)
print(f"\nTime used: {time_used:.3f} s")
print(f"Time requirement: ≤ 2.463 s")
print(f"Pass: {time_used <= 2.463}")

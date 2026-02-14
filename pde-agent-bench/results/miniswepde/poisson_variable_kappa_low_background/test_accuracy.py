import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
        }
    }
}

result = solve(case_spec)
u_grid = result["u"]

# Compute exact solution on the same grid
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u_exact_grid = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Compute error
error = np.abs(u_grid - u_exact_grid)
max_error = np.max(error)
mean_error = np.mean(error)
print(f"Max error: {max_error:.2e}")
print(f"Mean error: {mean_error:.2e}")

# Check against pass/fail criteria
if max_error <= 3.42e-04:
    print("Accuracy PASS")
else:
    print("Accuracy FAIL")

# Note: Time is checked by evaluator

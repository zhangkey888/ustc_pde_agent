import numpy as np
from solver import solve
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"}
        }
    }
}
result = solve(case_spec)
u_grid = result["u"]
nx, ny = u_grid.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print(f"Max error: {max_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
print(f"Required accuracy: error ≤ 2.14e-03")
print(f"Pass accuracy? {max_error <= 2.14e-03}")
print(f"Solver info: {result['solver_info']}")

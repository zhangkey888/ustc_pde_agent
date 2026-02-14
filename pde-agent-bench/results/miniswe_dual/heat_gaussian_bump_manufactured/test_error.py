import solver
import numpy as np

def u_exact(x, y, t):
    """Exact solution at time t"""
    return np.exp(-t) * np.exp(-40 * ((x - 0.5)**2 + (y - 0.5)**2))

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

print("Testing solver error...")
result = solver.solve(case_spec)

# Create grid for exact solution
nx = ny = 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Exact solution at t_end = 0.1
u_exact_grid = u_exact(X, Y, 0.1)

# Numerical solution
u_num = result["u"]

# Compute L2 error on the grid
error_grid = u_num - u_exact_grid
l2_error = np.sqrt(np.mean(error_grid**2))
max_error = np.max(np.abs(error_grid))

print(f"\nL2 error: {l2_error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Accuracy requirement: ≤ 2.49e-03")
print(f"L2 error meets requirement: {l2_error <= 2.49e-03}")
print(f"Max error meets requirement: {max_error <= 2.49e-03}")

# Check initial condition error
u0_exact = u_exact(X, Y, 0.0)
u0_num = result["u_initial"]
error_u0 = np.sqrt(np.mean((u0_num - u0_exact)**2))
print(f"\nInitial condition L2 error: {error_u0:.6e}")

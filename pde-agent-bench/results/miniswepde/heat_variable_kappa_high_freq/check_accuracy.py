import numpy as np
import solver

# Create case specification
case_spec = {
    "t_end": 0.1,
    "dt": 0.005,
    "scheme": "backward_euler"
}

result = solver.solve(case_spec)
u_grid = result["u"]

# Compute exact solution on the same 50x50 grid
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

# Exact solution at t=0.1: exp(-0.1)*sin(2πx)*sin(2πy)
u_exact = np.exp(-0.1) * np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)

# Compute L2 error on the grid (approximate)
dx = 1.0/(nx-1)
dy = 1.0/(ny-1)
error_grid = u_grid - u_exact
l2_error = np.sqrt(np.sum(error_grid**2) * dx * dy)
max_error = np.max(np.abs(error_grid))

print(f"L2 error on 50x50 grid: {l2_error:.6e}")
print(f"Max error on 50x50 grid: {max_error:.6e}")
print(f"Accuracy requirement: ≤ 1.22e-03")
print(f"L2 error meets requirement: {l2_error <= 1.22e-3}")
print(f"Max error meets requirement: {max_error <= 1.22e-3}")

# Also check initial condition
u0_grid = result["u_initial"]
u0_exact = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
error_u0 = np.max(np.abs(u0_grid - u0_exact))
print(f"\nInitial condition max error: {error_u0:.6e}")

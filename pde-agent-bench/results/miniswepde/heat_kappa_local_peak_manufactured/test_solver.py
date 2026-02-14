import numpy as np
from solver import solve

# Test case
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

result = solve(case_spec)
u_grid = result['u']
solver_info = result['solver_info']

# Compute exact solution at t=0.1 on the same grid
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Exact solution at t=0.1
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(2 * np.pi * Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 1.80e-03")
print(f"Time steps: {solver_info['n_steps']}")
print(f"Mesh resolution: {solver_info['mesh_resolution']}")
print(f"Total linear iterations: {solver_info['iterations']}")

# Check if error meets requirement
if max_error <= 1.80e-03:
    print("✓ Accuracy requirement met!")
else:
    print("✗ Accuracy requirement NOT met!")

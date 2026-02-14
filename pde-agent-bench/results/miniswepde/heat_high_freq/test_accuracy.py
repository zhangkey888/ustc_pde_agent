import numpy as np
from solver import solve

# Test case specification
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}

# Run solver
result = solve(case_spec)
u_grid = result["u"]
solver_info = result["solver_info"]

print(f"Mesh resolution: {solver_info['mesh_resolution']}")
print(f"Time taken: {solver_info['wall_time_sec']:.2f} seconds")

# Compute exact solution on the same grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution at t_end = 0.1
u_exact = np.exp(-0.1) * np.sin(4*np.pi*X) * np.sin(4*np.pi*Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.2e}")
print(f"Mean error: {mean_error:.2e}")
print(f"L2 error: {l2_error:.2e}")

# Check if meets accuracy requirement
accuracy_requirement = 8.89e-03
if l2_error <= accuracy_requirement:
    print(f"✓ PASS: L2 error {l2_error:.2e} ≤ {accuracy_requirement:.2e}")
else:
    print(f"✗ FAIL: L2 error {l2_error:.2e} > {accuracy_requirement:.2e}")

# Check time requirement
time_requirement = 26.841
if solver_info['wall_time_sec'] <= time_requirement:
    print(f"✓ PASS: Time {solver_info['wall_time_sec']:.2f}s ≤ {time_requirement}s")
else:
    print(f"✗ FAIL: Time {solver_info['wall_time_sec']:.2f}s > {time_requirement}s")

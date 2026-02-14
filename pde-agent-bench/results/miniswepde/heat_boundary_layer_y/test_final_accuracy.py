import numpy as np
from solver import solve

# Create test case
test_case = {
    "pde": {
        "time": {
            "t_end": 0.08,
            "dt": 0.008,
            "scheme": "backward_euler"
        }
    }
}

# Run solver
result = solve(test_case)
u_grid = result['u']
solver_info = result['solver_info']

print(f"Mesh resolution: {solver_info['mesh_resolution']}")
print(f"Element degree: {solver_info['element_degree']}")
print(f"Solver: {solver_info['ksp_type']}/{solver_info['pc_type']}")
print(f"Wall time: {solver_info['wall_time_sec']:.3f}s")
print(f"Linear iterations: {solver_info['iterations']}")
print(f"Time steps: {solver_info['n_steps']}, dt: {solver_info['dt']:.6f}")

# Compute exact solution on the same grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

def exact_solution(x, y, t):
    return np.exp(-t) * np.exp(5*y) * np.sin(np.pi*x)

t_end = 0.08
u_exact = exact_solution(X, Y, t_end)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"\nAccuracy metrics:")
print(f"Max error: {max_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
print(f"Required error: ≤ 1.06e-03")
print(f"Pass accuracy: {l2_error <= 1.06e-03}")
print(f"Pass time: {solver_info['wall_time_sec'] <= 14.32}")

# Check solution range
print(f"\nSolution range:")
print(f"u min/max: {u_grid.min():.3f}, {u_grid.max():.3f}")
print(f"u_exact min/max: {u_exact.min():.3f}, {u_exact.max():.3f}")

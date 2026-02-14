import numpy as np
import time
import solver

def exact_solution(x, y, t):
    """u_exact = exp(-t)*sin(pi*x)*sin(pi*y)"""
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Run solver and measure time
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

start_time = time.time()
result = solver.solve(case_spec)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.3f} seconds")
print(f"Time constraint: ≤ 12.519 seconds")
print(f"Pass time constraint: {execution_time <= 12.519}")

# Compute error on the 50x50 grid
u_computed = result['u']
nx, ny = u_computed.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

t_end = 0.1
u_exact_grid = exact_solution(X, Y, t_end)

# L2 error approximation (trapezoidal rule)
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
error_grid = u_computed - u_exact_grid
l2_error = np.sqrt(np.sum(error_grid**2) * dx * dy)

print(f"\nL2 error: {l2_error:.6e}")
print(f"Accuracy constraint: ≤ 1.42e-03")
print(f"Pass accuracy constraint: {l2_error <= 1.42e-03}")

# Also check max error
max_error = np.max(np.abs(error_grid))
print(f"Max error: {max_error:.6e}")

# Check solver info
info = result['solver_info']
print(f"\nSolver info:")
print(f"  mesh_resolution: {info['mesh_resolution']}")
print(f"  element_degree: {info['element_degree']}")
print(f"  ksp_type: {info['ksp_type']}")
print(f"  pc_type: {info['pc_type']}")
print(f"  iterations: {info['iterations']}")
print(f"  dt: {info['dt']}")
print(f"  n_steps: {info['n_steps']}")

if execution_time <= 12.519 and l2_error <= 1.42e-03:
    print("\n✓ All constraints satisfied!")
else:
    print("\n✗ Some constraints not satisfied.")

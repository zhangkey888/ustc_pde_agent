import numpy as np
import time
import sys
sys.path.insert(0, '.')
from solver import solve

# Test case
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    },
    "coefficients": {
        "kappa": 1.0
    }
}

# Run and time
start = time.time()
result = solve(case_spec)
end = time.time()
wall_time = end - start

# Check accuracy
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(result['u'] - u_exact)
max_error = np.max(error)

print(f"Wall time: {wall_time:.3f} seconds")
print(f"Max error: {max_error:.2e}")
print(f"Accuracy requirement (≤ 1.42e-03): {max_error <= 1.42e-03}")
print(f"Time requirement (≤ 12.519s): {wall_time <= 12.519}")
print(f"PASS ALL: {max_error <= 1.42e-03 and wall_time <= 12.519}")

# Check solver_info fields
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol', 
                   'iterations', 'dt', 'n_steps', 'time_scheme']
print("\nChecking solver_info fields:")
for field in required_fields:
    if field in result['solver_info']:
        print(f"  ✓ {field}: {result['solver_info'][field]}")
    else:
        print(f"  ✗ {field}: MISSING")

# Check for wall_time_sec (should NOT be there)
if 'wall_time_sec' in result['solver_info']:
    print(f"\nWARNING: wall_time_sec should not be in solver_info")

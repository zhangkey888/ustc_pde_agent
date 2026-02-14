import numpy as np
import time
from solver import solve

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
        }
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.3f} seconds")
print(f"Solution shape: {result['u'].shape}")
print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

solver_info = result['solver_info']
print("\nSolver info:")
for key, value in solver_info.items():
    print(f"  {key}: {value}")

# Check required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required:
    if field not in solver_info:
        print(f"ERROR: Missing required field {field}")
    else:
        print(f"OK: {field} present")

# Check no time fields (since elliptic)
time_fields = ["dt", "n_steps", "time_scheme"]
for field in time_fields:
    if field in solver_info:
        print(f"WARNING: Time field {field} present but PDE is elliptic")

# Accuracy check
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u_exact_grid = np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(result['u'] - u_exact_grid)
max_error = np.max(error)
print(f"\nMax error: {max_error:.2e}")
print(f"Accuracy requirement: ≤ 3.42e-04")
if max_error <= 3.42e-04:
    print("Accuracy: PASS")
else:
    print("Accuracy: FAIL")

if end - start <= 2.127:
    print("Time: PASS (within 2.127s)")
else:
    print(f"Time: FAIL ({end - start:.3f}s > 2.127s)")

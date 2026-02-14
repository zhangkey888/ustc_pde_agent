import time
import numpy as np

# Test 1: Default case
print("Test 1: Default case with time parameters")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

from solver import solve

start = time.time()
result1 = solve(case_spec1)
end = time.time()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"Total iterations: {result1['solver_info']['iterations']}")
print(f"Solution min/max: {result1['u'].min():.6f}, {result1['u'].max():.6f}")
print()

# Test 2: Smaller dt
print("Test 2: Smaller dt = 0.01")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

start = time.time()
result2 = solve(case_spec2)
end = time.time()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"Total iterations: {result2['solver_info']['iterations']}")
print(f"Solution min/max: {result2['u'].min():.6f}, {result2['u'].max():.6f}")
print()

# Test 3: No time in case_spec (should use defaults)
print("Test 3: No time parameters in case_spec")
case_spec3 = {
    "pde": {}
}

start = time.time()
result3 = solve(case_spec3)
end = time.time()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Mesh resolution: {result3['solver_info']['mesh_resolution']}")
print(f"Total iterations: {result3['solver_info']['iterations']}")
print(f"Solution min/max: {result3['u'].min():.6f}, {result3['u'].max():.6f}")
print()

# Check if solutions are similar (relative difference)
if result1 is not None and result2 is not None:
    diff = np.abs(result1['u'] - result2['u'])
    rel_diff = diff / (np.abs(result2['u']) + 1e-12)
    print(f"Max absolute difference between Test 1 and 2: {diff.max():.6e}")
    print(f"Max relative difference between Test 1 and 2: {rel_diff.max():.6e}")

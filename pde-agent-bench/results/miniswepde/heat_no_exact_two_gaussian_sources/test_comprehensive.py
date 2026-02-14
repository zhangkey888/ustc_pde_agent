import numpy as np
from solver import solve
import time

print("=== Comprehensive Solver Tests ===\n")

# Test 1: Default case
print("Test 1: Default case (t_end=0.1, dt=0.02)")
test_case1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

start = time.time()
result1 = solve(test_case1)
elapsed1 = time.time() - start

print(f"  Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"  Wall time: {elapsed1:.3f}s (solver reported: {result1['solver_info']['wall_time_sec']:.3f}s)")
print(f"  Solution range: [{result1['u'].min():.6f}, {result1['u'].max():.6f}]")
print(f"  Linear iterations: {result1['solver_info']['iterations']}")
print(f"  NaN values: {np.sum(np.isnan(result1['u']))}")
print()

# Test 2: Different time parameters
print("Test 2: Smaller time step (dt=0.01)")
test_case2 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

result2 = solve(test_case2)
print(f"  Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"  Time steps: {result2['solver_info']['n_steps']}")
print(f"  Wall time: {result2['solver_info']['wall_time_sec']:.3f}s")
print(f"  Solution range: [{result2['u'].min():.6f}, {result2['u'].max():.6f}]")
print()

# Test 3: Empty case spec (should use defaults)
print("Test 3: Empty case spec")
test_case3 = {}
result3 = solve(test_case3)
print(f"  Mesh resolution: {result3['solver_info']['mesh_resolution']}")
print(f"  Time steps: {result3['solver_info']['n_steps']} (dt={result3['solver_info']['dt']})")
print(f"  Wall time: {result3['solver_info']['wall_time_sec']:.3f}s")
print()

# Test 4: Check output format requirements
print("Test 4: Output format validation")
required_keys = ["u", "u_initial", "solver_info"]
solver_info_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                    "iterations", "dt", "n_steps", "time_scheme"]

all_good = True
for key in required_keys:
    if key not in result1:
        print(f"  ERROR: Missing key '{key}' in result")
        all_good = False

if all_good:
    for key in solver_info_keys:
        if key not in result1["solver_info"]:
            print(f"  ERROR: Missing key '{key}' in solver_info")
            all_good = False

if all_good:
    print("  PASS: All required keys present")
    
    # Check shapes
    if result1["u"].shape == (50, 50):
        print("  PASS: u has correct shape (50, 50)")
    else:
        print(f"  ERROR: u has wrong shape {result1['u'].shape}, expected (50, 50)")
        all_good = False
        
    if result1["u_initial"].shape == (50, 50):
        print("  PASS: u_initial has correct shape (50, 50)")
    else:
        print(f"  ERROR: u_initial has wrong shape {result1['u_initial'].shape}, expected (50, 50)")
        all_good = False

# Test 5: Time constraint check
print("\nTest 5: Time constraint verification")
max_time = 14.943
actual_time = result1['solver_info']['wall_time_sec']
if actual_time <= max_time:
    print(f"  PASS: Solver time {actual_time:.3f}s <= {max_time}s")
else:
    print(f"  FAIL: Solver time {actual_time:.3f}s > {max_time}s")

# Test 6: Check that initial condition is zero
print("\nTest 6: Initial condition check")
u0_max = np.abs(result1["u_initial"]).max()
if u0_max < 1e-10:
    print(f"  PASS: Initial condition is zero (max abs = {u0_max:.2e})")
else:
    print(f"  WARNING: Initial condition not exactly zero (max abs = {u0_max:.2e})")

print("\n=== All tests completed ===")

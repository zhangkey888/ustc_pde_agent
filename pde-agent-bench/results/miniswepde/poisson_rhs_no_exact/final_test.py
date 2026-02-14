import numpy as np
import time
from solver import solve

print("=== Final Comprehensive Test ===\n")

# Test 1: Basic functionality
print("Test 1: Basic solve")
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 0.5},
        "source": "sin(3*pi*x)*sin(2*pi*y)"
    }
}

start = time.time()
result = solve(case_spec)
elapsed = time.time() - start

print(f"  Solve time: {elapsed:.3f}s")
print(f"  Solution shape: {result['u'].shape}")
print(f"  Solver info keys: {list(result['solver_info'].keys())}")

# Test 2: Accuracy check
print("\nTest 2: Accuracy verification")
ref_data = np.load('oracle_output/reference.npz')
u_star = ref_data['u_star']
u_computed = result['u']
x_ref = ref_data['x']
y_ref = ref_data['y']

dx = x_ref[1] - x_ref[0]
dy = y_ref[1] - y_ref[0]
error = np.sqrt(np.sum((u_computed - u_star)**2) * dx * dy)
max_error = np.max(np.abs(u_computed - u_star))

print(f"  L2 error: {error:.2e} (requirement: ≤ 2.24e-02)")
print(f"  Max error: {max_error:.2e} (requirement: ≤ 2.24e-02)")
print(f"  Accuracy PASS: {error <= 2.24e-02 and max_error <= 2.24e-02}")

# Test 3: Time constraint check
print("\nTest 3: Time constraint verification")
print(f"  Solve time: {elapsed:.3f}s (requirement: ≤ 2.980s)")
print(f"  Time PASS: {elapsed <= 2.980}")

# Test 4: Solver info completeness
print("\nTest 4: Solver info completeness")
required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
missing_keys = [k for k in required_keys if k not in result['solver_info']]
print(f"  Required keys: {required_keys}")
print(f"  Missing keys: {missing_keys}")
print(f"  All required keys present: {len(missing_keys) == 0}")

# Test 5: Output values sanity check
print("\nTest 5: Output sanity check")
print(f"  Solution min/max: {np.min(result['u']):.6f}, {np.max(result['u']):.6f}")
print(f"  Solution finite: {np.all(np.isfinite(result['u']))}")
print(f"  Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"  Total iterations: {result['solver_info']['iterations']}")

print("\n=== All Tests Complete ===")

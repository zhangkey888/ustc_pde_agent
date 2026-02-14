import numpy as np
from solver import solve

print("=== Final Comprehensive Test ===")

# Test 1: Basic functionality
print("\n1. Testing basic solve...")
case_spec = {"pde": {"type": "elliptic"}}
result = solve(case_spec)
print(f"   Solution shape: {result['u'].shape}")
print(f"   Solver info keys: {list(result['solver_info'].keys())}")

# Test 2: Accuracy check
print("\n2. Testing accuracy...")
def exact_solution(x, y):
    return np.exp(2*x) * np.cos(np.pi * y)

u_grid = result["u"]
nx, ny = 50, 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
u_exact = exact_solution(X, Y)
error = np.max(np.abs(u_grid - u_exact))
print(f"   Max error: {error:.2e}")
print(f"   Required: ≤ 7.20e-05")
print(f"   Pass: {error <= 7.20e-05}")

# Test 3: Time check (rough)
print("\n3. Testing timing (single run)...")
import time
start = time.time()
result2 = solve(case_spec)
end = time.time()
elapsed = end - start
print(f"   Solve time: {elapsed:.3f} seconds")
print(f"   Required: ≤ 1.748 seconds")
print(f"   Pass: {elapsed <= 1.748}")

# Test 4: Check solver_info fields
print("\n4. Checking solver_info fields...")
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
info = result["solver_info"]
all_present = True
for field in required:
    if field not in info:
        print(f"   ✗ Missing: {field}")
        all_present = False
    else:
        print(f"   ✓ {field}: {info[field]}")
print(f"   All required fields present: {all_present}")

# Test 5: Check for NaN values
print("\n5. Checking for NaN values...")
has_nan = np.any(np.isnan(result["u"]))
print(f"   Has NaN: {has_nan}")
if has_nan:
    print("   WARNING: Solution contains NaN values!")

print("\n=== Test Complete ===")
if error <= 7.20e-05 and elapsed <= 1.748 and all_present and not has_nan:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed!")

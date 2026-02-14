import numpy as np
import time
from solver import solve

print("Testing edge cases...")

# Test 1: Default case (should work)
print("\nTest 1: Default parameters")
case1 = {"pde": {"epsilon": 0.02, "beta": [-8.0, 4.0]}}
start = time.time()
result1 = solve(case1)
print(f"  Time: {time.time() - start:.3f}s")
print(f"  Mesh: {result1['solver_info']['mesh_resolution']}")
print(f"  Iterations: {result1['solver_info']['iterations']}")

# Test 2: Missing epsilon
print("\nTest 2: Missing epsilon (should use default 0.02)")
case2 = {"pde": {"beta": [-8.0, 4.0]}}
result2 = solve(case2)
print(f"  Mesh: {result2['solver_info']['mesh_resolution']}")

# Test 3: Missing beta
print("\nTest 3: Missing beta (should use default [-8.0, 4.0])")
case3 = {"pde": {"epsilon": 0.02}}
result3 = solve(case3)
print(f"  Mesh: {result3['solver_info']['mesh_resolution']}")

# Test 4: Empty pde dict
print("\nTest 4: Empty pde dict (should use all defaults)")
case4 = {"pde": {}}
result4 = solve(case4)
print(f"  Mesh: {result4['solver_info']['mesh_resolution']}")

# Test 5: Different beta (still high Peclet)
print("\nTest 5: Different beta [10.0, -5.0]")
case5 = {"pde": {"epsilon": 0.02, "beta": [10.0, -5.0]}}
start = time.time()
result5 = solve(case5)
print(f"  Time: {time.time() - start:.3f}s")
print(f"  Mesh: {result5['solver_info']['mesh_resolution']}")

print("\nAll edge case tests completed!")

import numpy as np
from solver import solve

print("Test 1: Default case_spec")
result1 = solve({})
print(f"  dt used: {result1['solver_info']['dt']}")
print(f"  n_steps: {result1['solver_info']['n_steps']}")

print("\nTest 2: Partial time spec")
result2 = solve({"pde": {"time": {"t_end": 0.05}}})
print(f"  dt used: {result2['solver_info']['dt']}")
print(f"  n_steps: {result2['solver_info']['n_steps']}")

print("\nTest 3: Different dt")
result3 = solve({"pde": {"time": {"dt": 0.001, "t_end": 0.1}}})
print(f"  dt used: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")

# Check solver_info fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
            "iterations", "dt", "n_steps", "time_scheme"]
print(f"\nChecking required fields in solver_info:")
for field in required:
    if field in result1['solver_info']:
        print(f"  {field}: OK")
    else:
        print(f"  {field}: MISSING")

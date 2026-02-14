import numpy as np
from solver import solve

# Test 1: Default case_spec
print("Test 1: Default case_spec")
result1 = solve({})
print(f"  Shape: {result1['u'].shape}")
print(f"  Info: {result1['solver_info']}")

# Test 2: Case_spec with kappa
print("\nTest 2: Case_spec with kappa=2.0")
case_spec2 = {
    "pde": {
        "type": "elliptic",
        "coefficients": {"kappa": 2.0}
    }
}
result2 = solve(case_spec2)
print(f"  Shape: {result2['u'].shape}")
print(f"  Info: {result2['solver_info']}")

# Test 3: Malformed case_spec
print("\nTest 3: Malformed case_spec")
result3 = solve({"invalid": "data"})
print(f"  Shape: {result3['u'].shape}")
print(f"  Info: {result3['solver_info']}")

# Check that all required fields are present
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required_fields:
    if field not in result1['solver_info']:
        print(f"\nERROR: Missing field {field} in solver_info")
    else:
        print(f"\nOK: Field {field} present with value {result1['solver_info'][field]}")

print("\nAll tests completed.")

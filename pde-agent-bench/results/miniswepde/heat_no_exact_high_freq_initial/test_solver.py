import numpy as np
from solver import solve

# Test 1: Full case_spec
print("Test 1: Full case_spec")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}
result1 = solve(case_spec1)
print(f"  Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"  dt: {result1['solver_info']['dt']}")
print(f"  n_steps: {result1['solver_info']['n_steps']}")
print(f"  ksp_type: {result1['solver_info']['ksp_type']}")
print(f"  u shape: {result1['u'].shape}")
print(f"  u_initial shape: {result1['u_initial'].shape}")
print()

# Test 2: Missing time key (should use defaults)
print("Test 2: Missing time key")
case_spec2 = {
    "pde": {}
}
result2 = solve(case_spec2)
print(f"  Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"  dt: {result2['solver_info']['dt']}")
print(f"  n_steps: {result2['solver_info']['n_steps']}")
print(f"  t_end should be 0.12: {result2['solver_info']['dt'] * result2['solver_info']['n_steps']}")
print()

# Test 3: Only t_end provided
print("Test 3: Only t_end provided")
case_spec3 = {
    "pde": {
        "time": {
            "t_end": 0.1
        }
    }
}
result3 = solve(case_spec3)
print(f"  Mesh resolution: {result3['solver_info']['mesh_resolution']}")
print(f"  dt: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")
print(f"  t_end should be 0.1: {result3['solver_info']['dt'] * result3['solver_info']['n_steps']}")
print()

# Test 4: Check solver_info fields
print("Test 4: Check required solver_info fields")
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                   "iterations", "dt", "n_steps", "time_scheme"]
for field in required_fields:
    if field in result1['solver_info']:
        print(f"  ✓ {field}: {result1['solver_info'][field]}")
    else:
        print(f"  ✗ {field} missing")

# Test 5: Check solution values are finite
print("\nTest 5: Check solution values")
print(f"  u min: {np.min(result1['u'])}")
print(f"  u max: {np.max(result1['u'])}")
print(f"  u has NaN: {np.any(np.isnan(result1['u']))}")
print(f"  u has Inf: {np.any(np.isinf(result1['u']))}")

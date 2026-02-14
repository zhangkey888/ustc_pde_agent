import numpy as np
import solver

# Test 1: Full case specification
print("Test 1: Full case specification")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
result1 = solver.solve(case_spec1)
print(f"  u shape: {result1['u'].shape}")
print(f"  solver_info keys: {list(result1['solver_info'].keys())}")
print(f"  mesh_resolution: {result1['solver_info']['mesh_resolution']}")

# Test 2: Minimal case specification (missing time)
print("\nTest 2: Minimal case specification")
case_spec2 = {
    "pde": {}
}
result2 = solver.solve(case_spec2)
print(f"  u shape: {result2['u'].shape}")
print(f"  dt used: {result2['solver_info']['dt']}")
print(f"  n_steps: {result2['solver_info']['n_steps']}")

# Test 3: Partial time specification
print("\nTest 3: Partial time specification")
case_spec3 = {
    "pde": {
        "time": {
            "t_end": 0.2  # dt should default to 0.01
        }
    }
}
result3 = solver.solve(case_spec3)
print(f"  t_end requested: 0.2")
print(f"  dt used: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")

# Check that all required fields are present
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol', 
                   'iterations', 'dt', 'n_steps', 'time_scheme']
for field in required_fields:
    if field not in result1['solver_info']:
        print(f"WARNING: Missing field {field} in solver_info")

print("\nAll tests completed.")

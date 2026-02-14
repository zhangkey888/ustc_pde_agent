import numpy as np
from solver import solve

# Test 1: Default case
print("Test 1: Default case")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
result1 = solve(case_spec1)
print(f"  Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"  Solution shape: {result1['u'].shape}")
print(f"  Solver type: {result1['solver_info']['ksp_type']}/{result1['solver_info']['pc_type']}")
print(f"  Iterations: {result1['solver_info']['iterations']}")

# Test 2: Different time parameters
print("\nTest 2: Smaller time step")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}
result2 = solve(case_spec2)
print(f"  Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"  Solution shape: {result2['u'].shape}")
print(f"  Solver type: {result2['solver_info']['ksp_type']}/{result2['solver_info']['pc_type']}")
print(f"  Iterations: {result2['solver_info']['iterations']}")

# Test 3: Missing time parameters (should use defaults)
print("\nTest 3: Minimal case spec")
case_spec3 = {
    "pde": {
        "time": {}
    }
}
result3 = solve(case_spec3)
print(f"  Mesh resolution: {result3['solver_info']['mesh_resolution']}")
print(f"  Solution shape: {result3['u'].shape}")
print(f"  dt used: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")

print("\nAll tests completed!")

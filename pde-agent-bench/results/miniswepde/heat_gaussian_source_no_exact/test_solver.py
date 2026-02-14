import solver
import numpy as np

# Test 1: Full case_spec
print("Test 1: Full case_spec")
case1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}
result1 = solver.solve(case1)
print(f"  mesh_resolution: {result1['solver_info']['mesh_resolution']}")
print(f"  u shape: {result1['u'].shape}")
print(f"  u min/max: {result1['u'].min():.6f}, {result1['u'].max():.6f}")

# Test 2: Empty case_spec (should use defaults)
print("\nTest 2: Empty case_spec")
case2 = {}
result2 = solver.solve(case2)
print(f"  mesh_resolution: {result2['solver_info']['mesh_resolution']}")
print(f"  dt: {result2['solver_info']['dt']}")
print(f"  t_end computed: {result2['solver_info']['dt'] * result2['solver_info']['n_steps']}")

# Test 3: case_spec with different dt
print("\nTest 3: case_spec with dt=0.01")
case3 = {
    "pde": {
        "time": {
            "dt": 0.01,
            "t_end": 0.1
        }
    }
}
result3 = solver.solve(case3)
print(f"  dt: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")

# Test 4: case_spec with very small t_end
print("\nTest 4: case_spec with t_end=0.01")
case4 = {
    "pde": {
        "time": {
            "t_end": 0.01,
            "dt": 0.005
        }
    }
}
result4 = solver.solve(case4)
print(f"  dt: {result4['solver_info']['dt']}")
print(f"  n_steps: {result4['solver_info']['n_steps']}")

print("\nAll tests completed.")

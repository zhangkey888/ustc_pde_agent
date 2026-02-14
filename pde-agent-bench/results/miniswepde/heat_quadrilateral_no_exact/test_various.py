import solver
import numpy as np

# Test 1: Default case
print("Test 1: Default case")
case1 = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.03,
            "scheme": "backward_euler"
        }
    }
}
result1 = solver.solve(case1)
print(f"  Mesh: {result1['solver_info']['mesh_resolution']}, Steps: {result1['solver_info']['n_steps']}")

# Test 2: Different dt
print("\nTest 2: Smaller dt")
case2 = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
result2 = solver.solve(case2)
print(f"  Mesh: {result2['solver_info']['mesh_resolution']}, Steps: {result2['solver_info']['n_steps']}")

# Test 3: No time in case_spec (should use defaults)
print("\nTest 3: Empty case_spec")
case3 = {}
result3 = solver.solve(case3)
print(f"  Mesh: {result3['solver_info']['mesh_resolution']}, Steps: {result3['solver_info']['n_steps']}")

# Test 4: Partial time specification
print("\nTest 4: Partial time spec")
case4 = {
    "pde": {
        "time": {
            "t_end": 0.24  # dt should default to 0.03
        }
    }
}
result4 = solver.solve(case4)
print(f"  Mesh: {result4['solver_info']['mesh_resolution']}, Steps: {result4['solver_info']['n_steps']}, dt: {result4['solver_info']['dt']}")

print("\nAll tests passed!")

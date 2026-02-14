import solver
import numpy as np

# Test 1: Default case
print("Test 1: Default case_spec")
case1 = {
    "pde": {
        "time": {
            "t_end": 0.2,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}
result1 = solver.solve(case1)
print(f"  u shape: {result1['u'].shape}")
print(f"  solver_info: {result1['solver_info']}")
print(f"  u min/max: {result1['u'].min():.6f}, {result1['u'].max():.6f}")

# Test 2: Different time parameters
print("\nTest 2: Smaller dt")
case2 = {
    "pde": {
        "time": {
            "t_end": 0.2,
            "dt": 0.01,  # Smaller dt
            "scheme": "backward_euler"
        }
    }
}
result2 = solver.solve(case2)
print(f"  n_steps: {result2['solver_info']['n_steps']}")
print(f"  dt: {result2['solver_info']['dt']}")

# Test 3: Missing time key (should use defaults)
print("\nTest 3: Minimal case_spec")
case3 = {}
result3 = solver.solve(case3)
print(f"  dt: {result3['solver_info']['dt']}")
print(f"  t_end implied: {result3['solver_info']['dt'] * result3['solver_info']['n_steps']}")

# Test 4: Check initial condition
print("\nTest 4: Verify initial condition")
print(f"  u_initial shape: {result1['u_initial'].shape}")
print(f"  u_initial min/max: {result1['u_initial'].min():.6f}, {result1['u_initial'].max():.6f}")

# Test 5: Check convergence criteria
print("\nTest 5: Verify adaptive refinement")
print(f"  Mesh resolution used: {result1['solver_info']['mesh_resolution']}")
print(f"  Element degree: {result1['solver_info']['element_degree']}")

print("\nAll tests completed.")

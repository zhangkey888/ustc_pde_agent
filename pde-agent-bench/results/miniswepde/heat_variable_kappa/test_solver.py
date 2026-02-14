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
print(f"Error should be < 2.94e-03: {result1['solver_info'].get('final_error', 'N/A')}")
print(f"Wall time: {result1['solver_info']['wall_time']:.2f}s")
print(f"Mesh used: {result1['solver_info']['mesh_resolution']}")
print()

# Test 2: Different dt
print("Test 2: Smaller dt")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}
result2 = solve(case_spec2)
print(f"Error: {result2['solver_info'].get('final_error', 'N/A')}")
print(f"Wall time: {result2['solver_info']['wall_time']:.2f}s")
print(f"Mesh used: {result2['solver_info']['mesh_resolution']}")
print()

# Test 3: No time in case_spec (should use defaults)
print("Test 3: Minimal case_spec")
case_spec3 = {}
result3 = solve(case_spec3)
print(f"Error: {result3['solver_info'].get('final_error', 'N/A')}")
print(f"dt used: {result3['solver_info']['dt']}")
print(f"t_end computed from n_steps*dt: {result3['solver_info']['n_steps'] * result3['solver_info']['dt']}")
print()

# Check output shapes
print("Output shape checks:")
print(f"u shape: {result1['u'].shape} (should be (50, 50))")
print(f"u_initial shape: {result1['u_initial'].shape} (should be (50, 50))")
print(f"u min/max: {np.min(result1['u']):.3f}, {np.max(result1['u']):.3f}")
print(f"u_initial min/max: {np.min(result1['u_initial']):.3f}, {np.max(result1['u_initial']):.3f}")

import numpy as np
from solver import solve

# Test case
test_case = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

print("Testing adaptive mesh refinement...")
result = solve(test_case)
print(f"Final mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Wall time: {result['solver_info']['wall_time_sec']:.3f}s")
print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")

# Check if solution has reasonable values
if np.any(np.isnan(result['u'])):
    print("WARNING: Solution contains NaN values!")
else:
    print("Solution is valid (no NaN values).")

# Check time constraint
if result['solver_info']['wall_time_sec'] <= 14.943:
    print("PASS: Meets time constraint (< 14.943s)")
else:
    print(f"FAIL: Exceeds time constraint ({result['solver_info']['wall_time_sec']:.3f}s > 14.943s)")

print("\nSolver info:")
for key, value in result['solver_info'].items():
    print(f"  {key}: {value}")

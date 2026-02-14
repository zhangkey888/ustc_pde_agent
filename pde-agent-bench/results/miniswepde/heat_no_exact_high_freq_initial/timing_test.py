import time
import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

# Warm-up run
result = solve(case_spec)

# Timed run
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Solve time: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
print(f"Total linear iterations: {result['solver_info']['iterations']}")
print(f"Time per iteration: {(end_time - start_time) / result['solver_info']['iterations']:.6f} seconds")

# Check against constraints
if end_time - start_time <= 12.871:
    print("✓ Meets time constraint (< 12.871s)")
else:
    print("✗ Exceeds time constraint")

# The accuracy constraint is error ≤ 1.65e+05, but we don't have exact solution
# to compute error. The evaluator will handle that.

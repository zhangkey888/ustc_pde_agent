import time
from solver import solve

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Total iterations: {result['solver_info']['iterations']}")
print(f"Time per iteration: {(end_time - start_time) / max(1, result['solver_info']['iterations']):.6f} seconds")

# Check if within time limit
if end_time - start_time <= 17.829:
    print("✓ Within time constraint (≤ 17.829s)")
else:
    print(f"✗ Exceeds time constraint (>{17.829}s)")

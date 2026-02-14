import time
import numpy as np
from solver import solve

# Use the exact case specification from the problem
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

# Run multiple times to get stable timing
times = []
for i in range(3):
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    times.append(end - start)
    print(f"Run {i+1}: {times[-1]:.3f} seconds")

print(f"\nAverage time: {np.mean(times):.3f} seconds")
print(f"Std dev: {np.std(times):.3f} seconds")
print(f"Max time: {max(times):.3f} seconds")

# Check against constraint
time_limit = 41.146
if max(times) < time_limit:
    print(f"\n✓ PASS: All runs under time limit of {time_limit} seconds")
else:
    print(f"\n✗ FAIL: Some runs exceed time limit of {time_limit} seconds")

# Check solution properties
print(f"\nSolution properties:")
print(f"  Shape: {result['u'].shape}")
print(f"  Min: {result['u'].min():.6e}")
print(f"  Max: {result['u'].max():.6e}")
print(f"  Mean: {result['u'].mean():.6e}")

# Check solver_info
info = result['solver_info']
print(f"\nSolver info:")
for key, value in info.items():
    print(f"  {key}: {value}")

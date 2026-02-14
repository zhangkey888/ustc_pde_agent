import time
import numpy as np
from solver import solve

# Run multiple times to get stable timing
case_spec = {"pde": {"type": "elliptic"}}
times = []

for i in range(3):
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    times.append(end - start)
    if i == 0:
        print(f"First run solver info: {result['solver_info']}")
        print(f"Solution shape: {result['u'].shape}")

avg_time = np.mean(times)
print(f"\nAverage solve time over {len(times)} runs: {avg_time:.4f} seconds")
print(f"Time requirement: ≤ 1.748 seconds")
print(f"Pass: {avg_time <= 1.748}")

# Verify solver_info has all required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
info = result['solver_info']
print(f"\nChecking required fields in solver_info:")
for field in required:
    if field in info:
        print(f"  ✓ {field}: {info[field]}")
    else:
        print(f"  ✗ {field} missing!")

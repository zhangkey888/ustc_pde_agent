import time
import numpy as np
from solver import solve

# Test case
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

# Run multiple times to get average
n_runs = 3
times = []
for i in range(n_runs):
    start_time = time.time()
    result = solve(case_spec)
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.3f} seconds")

avg_time = np.mean(times)
print(f"\nAverage time: {avg_time:.3f} seconds")
print(f"Time requirement: ≤ 16.635 seconds")

if avg_time <= 16.635:
    print("✓ Time requirement met!")
else:
    print("✗ Time requirement NOT met!")

# Also check accuracy again
u_grid = result['u']
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print(f"\nMax error: {max_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 1.80e-03")

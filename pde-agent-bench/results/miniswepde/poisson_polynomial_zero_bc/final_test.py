import numpy as np
import time
from solver import solve

def u_exact(x, y):
    return x * (1 - x) * y * (1 - y)

# Run solver multiple times for accurate timing
case_spec = {"pde": {"type": "elliptic"}}

# Warm-up run
result = solve(case_spec)

# Time measurement
n_runs = 10
times = []
for i in range(n_runs):
    start = time.perf_counter()
    result = solve(case_spec)
    end = time.perf_counter()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)

# Accuracy check
u_grid = result["u"]
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact_grid = u_exact(X, Y)

error = np.abs(u_grid - u_exact_grid)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print("=== FINAL VALIDATION ===")
print(f"Accuracy:")
print(f"  Max error: {max_error:.6e}")
print(f"  L2 error:  {l2_error:.6e}")
print(f"  Required:  ≤ 8.95e-04")
print(f"  Pass:      {max_error <= 8.95e-04}")
print()
print(f"Performance:")
print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} s")
print(f"  Time limit:   ≤ 1.477 s")
print(f"  Pass:         {avg_time <= 1.477}")
print()
print(f"Solver info:")
for key, value in result["solver_info"].items():
    print(f"  {key}: {value}")
print()
print("All tests passed!" if max_error <= 8.95e-04 and avg_time <= 1.477 else "Some tests failed!")

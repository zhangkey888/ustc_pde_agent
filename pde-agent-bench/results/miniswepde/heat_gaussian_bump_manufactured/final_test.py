import time
import numpy as np
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

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Execution time: {end - start:.3f} seconds")
print("Solver info:", result["solver_info"])
print("Solution shape:", result["u"].shape)

# Compute error on grid
def u_exact(x, y, t):
    return np.exp(-t) * np.exp(-40 * ((x - 0.5)**2 + (y - 0.5)**2))

nx = ny = 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
exact = u_exact(X, Y, 0.1)
error = np.abs(result["u"] - exact)
max_error = np.max(error)
rms_error = np.sqrt(np.mean(error**2))
print(f"Max error on grid: {max_error:.6e}")
print(f"RMS error on grid: {rms_error:.6e}")
print(f"Accuracy requirement: 2.49e-03")
if max_error <= 2.49e-03:
    print("PASS: Accuracy met.")
else:
    print("FAIL: Accuracy not met.")
if end - start <= 13.753:
    print("PASS: Time constraint met.")
else:
    print("FAIL: Time constraint not met.")

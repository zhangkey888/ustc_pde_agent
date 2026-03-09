import numpy as np
import time

# Build a case_spec matching the problem description
case_spec = {
    "pde": {
        "parameters": {
            "epsilon": 0.01,
            "beta": [12.0, 0.0],
        }
    },
    "domain": {
        "bounds": [[0, 1], [0, 1]]
    },
    "output": {
        "nx": 50,
        "ny": 50
    }
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result["u"]
print(f"Solution shape: {u.shape}")
print(f"NaN count: {np.isnan(u).sum()}")
print(f"Solution range: [{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")

# Exact solution on 50x50 grid
xs = np.linspace(0, 1, 50)
ys = np.linspace(0, 1, 50)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
u_exact = np.exp(3.0 * XX) * np.sin(np.pi * YY)

diff = u - u_exact
rms_error = np.sqrt(np.nanmean(diff**2))
max_error = np.nanmax(np.abs(diff))
print(f"RMS error: {rms_error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Wall time: {elapsed:.3f}s")
print(f"Solver info: {result['solver_info']}")
print(f"\nTarget: error <= 1.63e-04, time <= 3.517s")
print(f"PASS accuracy: {rms_error <= 1.63e-4}")
print(f"PASS time: {elapsed <= 3.517}")

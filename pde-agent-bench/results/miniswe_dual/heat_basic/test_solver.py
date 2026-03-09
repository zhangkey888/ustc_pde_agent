import numpy as np
import time

from solver import solve

case_spec = {
    "pde": {
        "type": "heat",
        "coefficients": {"kappa": 1.0},
        "time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
    },
    "domain": {"type": "unit_square"},
    "output": {"nx": 50, "ny": 50},
}

t0 = time.perf_counter()
result = solve(case_spec)
elapsed = time.perf_counter() - t0

u_grid = result["u"]
info = result["solver_info"]

t_end = 0.1
nx, ny = 50, 50
xg = np.linspace(0, 1, nx)
yg = np.linspace(0, 1, ny)
X, Y = np.meshgrid(xg, yg, indexing='ij')
u_exact = np.exp(-t_end) * np.sin(np.pi * X) * np.sin(np.pi * Y)

l2_error = np.sqrt(np.mean((u_grid - u_exact) ** 2))
linf_error = np.max(np.abs(u_grid - u_exact))

print(f"Wall time: {elapsed:.3f} s")
print(f"L2 error:  {l2_error:.6e}")
print(f"Linf error: {linf_error:.6e}")
print(f"Solver info: {info}")
print(f"NaN count: {np.isnan(u_grid).sum()}")
print(f"Pass accuracy: {l2_error <= 1.42e-3}")
print(f"Pass time: {elapsed <= 9.727}")

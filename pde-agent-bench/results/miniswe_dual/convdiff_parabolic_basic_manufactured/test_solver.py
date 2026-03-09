import numpy as np
import time

case_spec = {
    "pde": {
        "type": "convection_diffusion",
        "parameters": {
            "epsilon": 0.1,
            "beta": [1.0, 0.5],
        },
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler",
        }
    },
    "domain": {
        "type": "unit_square",
    }
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Solution shape: {u_grid.shape}")
print(f"Wall time: {elapsed:.3f}s")
print(f"Solver info: {result['solver_info']}")

nx, ny = 50, 50
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
t_end = 0.1
u_exact = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)

diff = u_grid - u_exact
l2_err = np.sqrt(np.mean(diff**2))
linf_err = np.max(np.abs(diff))
rel_l2 = l2_err / np.sqrt(np.mean(u_exact**2))

print(f"L2 error: {l2_err:.6e}")
print(f"Linf error: {linf_err:.6e}")
print(f"Relative L2 error: {rel_l2:.6e}")
print(f"Target error: <= 8.92e-03")
print(f"Target time: <= 5.213s")
print(f"PASS accuracy: {l2_err <= 8.92e-3}")
print(f"PASS time: {elapsed <= 5.213}")

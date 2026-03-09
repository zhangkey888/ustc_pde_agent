import numpy as np
import time

case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "time": {
            "t_end": 0.2,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    },
    "params": {
        "epsilon": 1.0,
        "mesh_resolution": 48,
        "element_degree": 2,
    },
    "output": {
        "nx": 60,
        "ny": 60
    }
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Solution shape: {u_grid.shape}")
print(f"Wall time: {elapsed:.2f}s")
print(f"Solver info: {result['solver_info']}")

# Compute exact solution at t=0.2 on the same grid
nx, ny = 60, 60
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
t_end = 0.2
u_exact = np.exp(-t_end) * 0.2 * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)

error = np.sqrt(np.mean((u_grid - u_exact)**2))
linf_error = np.max(np.abs(u_grid - u_exact))
print(f"L2 error (grid): {error:.6e}")
print(f"Linf error (grid): {linf_error:.6e}")
print(f"Target: error <= 2.99e-03")
print(f"PASS" if error <= 2.99e-3 else "FAIL")

import time
import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "type": "navier_stokes",
        "viscosity": 0.01,
    },
    "domain": {
        "type": "unit_square",
        "dim": 2,
    },
    "manufactured_solution": {
        "u": ["0.2*pi*cos(pi*y)*sin(2*pi*x)", "-0.4*pi*cos(2*pi*x)*sin(pi*y)"],
        "p": "0",
    }
}

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Wall time: {elapsed:.3f}s")
print(f"u_grid shape: {u_grid.shape}")
print(f"u_grid min: {np.nanmin(u_grid):.6e}, max: {np.nanmax(u_grid):.6e}")
print(f"NaN count: {np.isnan(u_grid).sum()}")
print(f"Solver info: {result['solver_info']}")

# Compute exact velocity magnitude on same grid
nx, ny = 50, 50
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
pi = np.pi
ux_exact = 0.2 * pi * np.cos(pi * YY) * np.sin(2 * pi * XX)
uy_exact = -0.4 * pi * np.cos(2 * pi * XX) * np.sin(pi * YY)
vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

# L2-like error
diff = u_grid - vel_mag_exact
l2_err = np.sqrt(np.nanmean(diff**2))
linf_err = np.nanmax(np.abs(diff))
print(f"L2 error (grid): {l2_err:.6e}")
print(f"Linf error (grid): {linf_err:.6e}")

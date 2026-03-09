import numpy as np
import time

case_spec = {
    "pde": {
        "type": "navier_stokes",
        "viscosity": 2.0,
        "manufactured_solution": {
            "u": ["0.5*pi*cos(pi*y)*sin(pi*x)", "-0.5*pi*cos(pi*x)*sin(pi*y)"],
            "p": "cos(pi*x) + cos(pi*y)"
        }
    },
    "domain": {"type": "unit_square", "bounds": [[0,1],[0,1]]},
    "output": {"field": "velocity_magnitude", "grid": [50, 50]}
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Solution shape: {u_grid.shape}")
print(f"Wall time: {elapsed:.3f}s")
print(f"Solver info: {result['solver_info']}")

# Compute exact velocity magnitude on same grid
nx, ny = 50, 50
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

ux_exact = 0.5 * np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
uy_exact = -0.5 * np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

diff = u_grid - vel_mag_exact
l2_err = np.sqrt(np.mean(diff**2))
l2_exact = np.sqrt(np.mean(vel_mag_exact**2))
rel_err = l2_err / l2_exact if l2_exact > 0 else l2_err

print(f"L2 absolute error: {l2_err:.6e}")
print(f"L2 relative error: {rel_err:.6e}")
print(f"Max absolute error: {np.max(np.abs(diff)):.6e}")
print(f"NaN count: {np.sum(np.isnan(u_grid))}")

if l2_err <= 1e-6:
    print("ACCURACY: PASS")
else:
    print("ACCURACY: FAIL")

if elapsed <= 5.745:
    print("TIME: PASS")
else:
    print("TIME: FAIL")

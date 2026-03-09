import numpy as np
import time

# Create a minimal case_spec
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
        }
    },
    "domain": {"type": "unit_square"},
    "manufactured_solution": "sin(pi*x)*sin(pi*y)"
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
print(f"Shape: {u_grid.shape}")
print(f"Time: {elapsed:.3f}s")
print(f"NaN count: {np.isnan(u_grid).sum()}")
print(f"Min: {np.nanmin(u_grid):.6f}, Max: {np.nanmax(u_grid):.6f}")

# Compute exact solution on same grid
nx_out, ny_out = 50, 50
xs = np.linspace(0.0, 1.0, nx_out)
ys = np.linspace(0.0, 1.0, ny_out)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)

error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f"L2 (RMS) error: {error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Solver info: {result['solver_info']}")

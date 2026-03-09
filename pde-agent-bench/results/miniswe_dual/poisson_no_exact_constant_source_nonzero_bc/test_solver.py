import time
import numpy as np

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0},
        "source": {"value": 1.0},
        "boundary_conditions": [
            {"type": "dirichlet", "value": 0.0}
        ],
    },
    "domain": {
        "extents": [[0.0, 1.0], [0.0, 1.0]],
    },
    "output": {
        "nx": 50,
        "ny": 50,
    },
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result["u"]
info = result["solver_info"]

print(f"Wall time: {elapsed:.3f}s")
print(f"Solution shape: {u_grid.shape}")
print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
print(f"NaN count: {np.isnan(u_grid).sum()}")
print(f"Solver info: {info}")
print(f"Max value (expected ~0.0737): {np.nanmax(u_grid):.6f}")

# Test with nonzero BC
case_spec2 = dict(case_spec)
case_spec2["pde"] = dict(case_spec["pde"])
case_spec2["pde"]["boundary_conditions"] = [{"type": "dirichlet", "value": 1.0}]

t0 = time.time()
result2 = solve(case_spec2)
elapsed2 = time.time() - t0
print(f"\nNonzero BC test: time={elapsed2:.3f}s, range=[{np.nanmin(result2['u']):.6f}, {np.nanmax(result2['u']):.6f}]")

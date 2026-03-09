import time
import numpy as np

case_spec = {
    "pde": {
        "type": "poisson",
        "source": "exp(-250*((x-0.4)**2 + (y-0.6)**2))",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 50*exp(-150*((x-0.5)**2 + (y-0.5)**2))"}
        }
    },
    "domain": {"type": "unit_square"},
    "boundary_conditions": {"type": "dirichlet", "value": 0.0}
}

from solver import solve

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result["u"]
print(f"Time: {elapsed:.3f}s")
print(f"Shape: {u.shape}")
print(f"Max: {np.nanmax(u):.8e}")
print(f"Min: {np.nanmin(u):.8e}")
print(f"NaN count: {np.isnan(u).sum()}")
print(f"Solver info: {result['solver_info']}")

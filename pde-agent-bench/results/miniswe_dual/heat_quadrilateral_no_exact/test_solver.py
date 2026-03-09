import time
import numpy as np

case_spec = {
    "pde": {
        "type": "heat",
        "source_term": 1.0,
        "initial_condition": 0.0,
        "coefficients": {"kappa": 1.0},
        "time": {"t_end": 0.12, "dt": 0.03, "scheme": "backward_euler"},
        "domain": {"type": "unit_square", "cell_type": "quadrilateral"}
    },
    "output": {"nx": 50, "ny": 50}
}

from solver import solve

start = time.time()
result = solve(case_spec)
elapsed = time.time() - start

u = result["u"]
info = result["solver_info"]

print(f"Solution shape: {u.shape}")
print(f"Solution min: {u.min():.6f}, max: {u.max():.6f}")
print(f"Solution mean: {u.mean():.6f}")
print(f"Wall time: {elapsed:.3f}s")
print(f"Solver info: {info}")
print(f"Time limit: 29.708s -> {'PASS' if elapsed < 29.708 else 'FAIL'}")

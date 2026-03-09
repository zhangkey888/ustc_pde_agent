import numpy as np
import time
from solver import solve

# Test with direct pde spec (no oracle_config wrapper)
case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "pde_params": {
            "epsilon": 0.05,
            "reaction": {"type": "allen_cahn", "lambda": 2.0}
        },
        "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
        "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
        "time": {
            "t0": 0.0,
            "t_end": 0.2,
            "dt": 0.005,
            "scheme": "backward_euler",
        },
    },
}

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result['u']
print(f"Direct spec: {elapsed:.2f}s, shape={u.shape}, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
print(f"NaN count: {np.isnan(u).sum()}")

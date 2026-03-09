import numpy as np
import sys
sys.path.insert(0, '.')
from solver import _solve_at_resolution

case_spec = {
    "pde": {
        "type": "convection_diffusion",
        "params": {
            "epsilon": 0.02,
            "beta": [8.0, 3.0],
        },
        "bcs": {},
    },
    "domain": {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
    },
    "output": {
        "nx": 50,
        "ny": 50,
    },
}

import time

# Test various resolutions and degrees
for deg in [1, 2]:
    for N in [64, 128, 256]:
        t0 = time.time()
        u_grid, info, l2 = _solve_at_resolution(
            N, deg, 0.02, [8.0, 3.0],
            [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec
        )
        elapsed = time.time() - t0
        print(f"deg={deg}, N={N}: max={np.nanmax(u_grid):.6f}, l2={l2:.6f}, time={elapsed:.2f}s")

# Compare P1 N=128 vs P2 N=128
u1, _, _ = _solve_at_resolution(128, 1, 0.02, [8.0, 3.0], [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec)
u2, _, _ = _solve_at_resolution(128, 2, 0.02, [8.0, 3.0], [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec)
diff = np.sqrt(np.nanmean((u1 - u2)**2))
print(f"\nRMS diff P1 vs P2 at N=128: {diff:.6e}")
print(f"Max diff: {np.nanmax(np.abs(u1 - u2)):.6e}")

import numpy as np
import sys, time
sys.path.insert(0, '.')
from solver import _solve_at_resolution

case_spec = {
    "pde": {
        "type": "convection_diffusion",
        "params": {"epsilon": 0.02, "beta": [8.0, 3.0]},
        "bcs": {},
    },
    "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
    "output": {"nx": 50, "ny": 50},
}

# Test P1 at different resolutions
for N in [64, 128]:
    t0 = time.time()
    u_grid, info, l2 = _solve_at_resolution(
        N, 1, 0.02, [8.0, 3.0],
        [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec
    )
    elapsed = time.time() - t0
    print(f"P1 N={N}: max={np.nanmax(u_grid):.6f}, l2={l2:.6f}, time={elapsed:.2f}s")

# Test P2 at N=64
t0 = time.time()
u_grid2, info2, l2_2 = _solve_at_resolution(
    64, 2, 0.02, [8.0, 3.0],
    [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec
)
elapsed = time.time() - t0
print(f"P2 N=64: max={np.nanmax(u_grid2):.6f}, l2={l2_2:.6f}, time={elapsed:.2f}s")

# Compare P1 N=128 vs P2 N=64
u1, _, _ = _solve_at_resolution(128, 1, 0.02, [8.0, 3.0], [0.0, 1.0], [0.0, 1.0], 50, 50, case_spec)
diff = np.sqrt(np.nanmean((u1 - u_grid2)**2))
print(f"\nRMS diff P1@128 vs P2@64: {diff:.6e}")

import numpy as np
import time
from solver import solve

case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
        "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
        "epsilon": 0.01,
        "reaction_type": "allen_cahn",
        "reaction_params": {},
        "time": {
            "t_end": 0.2,
            "dt": 0.005,
            "scheme": "backward_euler",
        },
        "domain": {
            "type": "unit_square",
        },
        "boundary_conditions": {
            "type": "dirichlet",
            "value": 0.0,
        },
    },
}

# Test with N=128, deg=2
case_spec["mesh_resolution"] = 128
case_spec["element_degree"] = 2
t0 = time.time()
res128 = solve(case_spec)
elapsed = time.time() - t0
print(f"N=128,deg=2: range=[{np.nanmin(res128['u']):.8f}, {np.nanmax(res128['u']):.8f}], time={elapsed:.2f}s")

# Compare with N=96
case_spec["mesh_resolution"] = 96
t0 = time.time()
res96 = solve(case_spec)
elapsed96 = time.time() - t0
print(f"N=96,deg=2: range=[{np.nanmin(res96['u']):.8f}, {np.nanmax(res96['u']):.8f}], time={elapsed96:.2f}s")

diff = np.abs(res96['u'] - res128['u'])
print(f"Max diff N=96 vs N=128: {np.nanmax(diff):.6e}")
print(f"L2 diff: {np.sqrt(np.nanmean(diff**2)):.6e}")

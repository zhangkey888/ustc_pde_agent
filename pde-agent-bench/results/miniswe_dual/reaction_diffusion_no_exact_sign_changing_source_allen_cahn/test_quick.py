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

# Test with N=64, deg=2
case_spec["mesh_resolution"] = 64
case_spec["element_degree"] = 2
t0 = time.time()
res64 = solve(case_spec)
elapsed = time.time() - t0
print(f"N=64,deg=2: range=[{np.nanmin(res64['u']):.8f}, {np.nanmax(res64['u']):.8f}], time={elapsed:.2f}s")

# Test with N=96, deg=2
case_spec["mesh_resolution"] = 96
t0 = time.time()
res96 = solve(case_spec)
elapsed = time.time() - t0
print(f"N=96,deg=2: range=[{np.nanmin(res96['u']):.8f}, {np.nanmax(res96['u']):.8f}], time={elapsed:.2f}s")

diff = np.abs(res64['u'] - res96['u'])
print(f"Max diff N=64 vs N=96: {np.nanmax(diff):.6e}")
print(f"L2 diff: {np.sqrt(np.nanmean(diff**2)):.6e}")

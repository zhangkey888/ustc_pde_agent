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
    },
}

# N=200, P1
case_spec["mesh_resolution"] = 200
case_spec["element_degree"] = 1
t0 = time.time()
res200 = solve(case_spec)
elapsed = time.time() - t0
print(f"N=200,P1: range=[{np.nanmin(res200['u']):.8f}, {np.nanmax(res200['u']):.8f}], time={elapsed:.2f}s")

# N=256, P1
case_spec["mesh_resolution"] = 256
t0 = time.time()
res256 = solve(case_spec)
elapsed = time.time() - t0
print(f"N=256,P1: range=[{np.nanmin(res256['u']):.8f}, {np.nanmax(res256['u']):.8f}], time={elapsed:.2f}s")

diff = np.abs(res200['u'] - res256['u'])
print(f"Max diff N=200 vs N=256: {np.nanmax(diff):.6e}, L2 diff: {np.sqrt(np.nanmean(diff**2)):.6e}")

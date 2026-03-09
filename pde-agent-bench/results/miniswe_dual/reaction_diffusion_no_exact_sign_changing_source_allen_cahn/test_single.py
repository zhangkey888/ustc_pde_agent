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

case_spec["mesh_resolution"] = 200
case_spec["element_degree"] = 1
t0 = time.time()
res = solve(case_spec)
elapsed = time.time() - t0
print(f"N=200,P1: range=[{np.nanmin(res['u']):.8f}, {np.nanmax(res['u']):.8f}], time={elapsed:.2f}s")

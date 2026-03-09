import numpy as np
import time
from solver import solve

cs = {
    "oracle_config": {
        "pde": {
            "type": "reaction_diffusion",
            "pde_params": {"epsilon": 0.05, "reaction": {"type": "allen_cahn", "lambda": 2.0}},
            "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
            "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"},
        },
        "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
        "output": {"grid": {"bbox": [0, 1, 0, 1], "nx": 70, "ny": 70}},
    },
    "mesh_resolution": 140,
    "element_degree": 1,
}

t0 = time.time()
r1 = solve(cs)
e1 = time.time() - t0
u1 = r1['u']
print(f"N=140,P1: {e1:.2f}s, range=[{np.nanmin(u1):.6f}, {np.nanmax(u1):.6f}]")

cs["mesh_resolution"] = 200
t0 = time.time()
r2 = solve(cs)
e2 = time.time() - t0
u2 = r2['u']
print(f"N=200,P1: {e2:.2f}s, range=[{np.nanmin(u2):.6f}, {np.nanmax(u2):.6f}]")

diff = u2 - u1
rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(u2**2))
print(f"rel_L2 diff: {rel_l2:.6e}, max_diff: {np.max(np.abs(diff)):.6e}")

import numpy as np
import time
from solver import solve

case_spec_base = {
    "oracle_config": {
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
        "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
        "output": {
            "format": "npz",
            "field": "scalar",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 70, "ny": 70}
        },
    },
}

results = {}
for N in [100, 140, 200]:
    cs = dict(case_spec_base)
    cs["mesh_resolution"] = N
    cs["element_degree"] = 1
    
    t0 = time.time()
    result = solve(cs)
    elapsed = time.time() - t0
    
    u = result['u']
    results[N] = u
    print(f"N={N}: time={elapsed:.2f}s, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}], L2={np.sqrt(np.mean(u**2)):.8f}")

# Compare pairs
for N in [100, 140]:
    diff = results[200] - results[N]
    rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(results[200]**2))
    max_diff = np.max(np.abs(diff))
    print(f"N={N} vs N=200: rel_L2={rel_l2:.6e}, max_diff={max_diff:.6e}")

# Also test with P2
cs = dict(case_spec_base)
cs["mesh_resolution"] = 140
cs["element_degree"] = 2

t0 = time.time()
result = solve(cs)
elapsed = time.time() - t0
u_p2 = result['u']
print(f"N=140,P2: time={elapsed:.2f}s, range=[{np.nanmin(u_p2):.6f}, {np.nanmax(u_p2):.6f}], L2={np.sqrt(np.mean(u_p2**2)):.8f}")

diff = u_p2 - results[200]
rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(results[200]**2))
print(f"N=140,P2 vs N=200,P1: rel_L2={rel_l2:.6e}")

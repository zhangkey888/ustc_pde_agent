import numpy as np
import time
from solver import solve

# Test with different resolutions
for N in [80, 160, 256]:
    for dt in [0.005, 0.0025]:
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
                    "dt": dt,
                    "scheme": "backward_euler",
                },
            },
            "mesh_resolution": N,
            "element_degree": 1,
        }
        
        t0 = time.time()
        result = solve(case_spec)
        elapsed = time.time() - t0
        
        u = result['u']
        print(f"N={N}, dt={dt}: time={elapsed:.2f}s, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}], L2norm={np.sqrt(np.nanmean(u**2)):.6f}")
        
        if N == 160 and dt == 0.005:
            u_ref = u.copy()
        elif 'u_ref' in dir():
            diff = np.nanmax(np.abs(u - u_ref))
            print(f"  Max diff from ref (N=160,dt=0.005): {diff:.6e}")

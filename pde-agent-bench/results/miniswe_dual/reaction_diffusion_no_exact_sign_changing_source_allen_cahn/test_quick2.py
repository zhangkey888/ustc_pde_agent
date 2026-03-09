import numpy as np
import time
from solver import solve

# Test with default settings
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
    "mesh_resolution": 80,
    "element_degree": 1,
}

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result['u']
print(f"N=80: time={elapsed:.2f}s, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
u_80 = u.copy()

# Now N=160
case_spec["mesh_resolution"] = 160
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result['u']
diff = np.nanmax(np.abs(u - u_80))
print(f"N=160: time={elapsed:.2f}s, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}], max_diff_from_80={diff:.6e}")

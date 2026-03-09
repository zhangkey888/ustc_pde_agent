import numpy as np
import time
from solver import solve

# Test convergence
case_spec_base = {
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

results = {}
for N in [32, 48, 64, 96]:
    cs = dict(case_spec_base)
    cs["mesh_resolution"] = N
    cs["element_degree"] = 2
    t0 = time.time()
    res = solve(cs)
    elapsed = time.time() - t0
    results[N] = res["u"]
    print(f"N={N}: range=[{np.nanmin(res['u']):.8f}, {np.nanmax(res['u']):.8f}], time={elapsed:.2f}s")

# Compare solutions
for N in [32, 48, 64]:
    diff = np.abs(results[N] - results[96])
    max_diff = np.nanmax(diff)
    l2_diff = np.sqrt(np.nanmean(diff**2))
    print(f"N={N} vs N=96: max_diff={max_diff:.6e}, L2_diff={l2_diff:.6e}")

# Also test with smaller dt
print("\n--- Testing dt sensitivity ---")
for dt_test in [0.01, 0.005, 0.002, 0.001]:
    cs = dict(case_spec_base)
    cs["pde"] = dict(case_spec_base["pde"])
    cs["pde"]["time"] = {"t_end": 0.2, "dt": dt_test, "scheme": "backward_euler"}
    cs["mesh_resolution"] = 64
    cs["element_degree"] = 2
    t0 = time.time()
    res = solve(cs)
    elapsed = time.time() - t0
    print(f"dt={dt_test}: range=[{np.nanmin(res['u']):.8f}, {np.nanmax(res['u']):.8f}], time={elapsed:.2f}s")

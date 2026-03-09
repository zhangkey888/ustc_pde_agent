import numpy as np
import time
import sys
sys.path.insert(0, '.')
from solver import solve

case_spec = {
    "pde": {
        "type": "convection_diffusion",
        "parameters": {
            "epsilon": 0.02,
            "beta": [6.0, 2.0],
        },
        "source": "exp(-200*((x-0.3)**2 + (y-0.7)**2))*exp(-t)",
        "initial_condition": "0.0",
        "time": {
            "t_end": 0.1,
            "dt": 0.02,
            "scheme": "backward_euler",
        },
        "boundary_conditions": {},
    },
    "domain": {
        "type": "unit_square",
    },
}

# Test with current settings
t0 = time.time()
result1 = solve(case_spec)
t1 = time.time()
u1 = result1["u"]
print(f"N=80, dt=0.02: max={u1.max():.6e}, L2={np.sqrt(np.sum(u1**2)/u1.size):.6e}, time={t1-t0:.2f}s")

# Now test with finer dt
case_spec2 = dict(case_spec)
case_spec2["pde"] = dict(case_spec["pde"])
case_spec2["pde"]["time"] = {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}
t0 = time.time()
result2 = solve(case_spec2)
t1 = time.time()
u2 = result2["u"]
print(f"N=80, dt=0.01: max={u2.max():.6e}, L2={np.sqrt(np.sum(u2**2)/u2.size):.6e}, time={t1-t0:.2f}s")

# Compare
diff = np.sqrt(np.sum((u1 - u2)**2) / u1.size)
print(f"RMS difference between dt=0.02 and dt=0.01: {diff:.6e}")
rel_diff = diff / (np.sqrt(np.sum(u2**2)/u2.size) + 1e-15)
print(f"Relative difference: {rel_diff:.6e}")

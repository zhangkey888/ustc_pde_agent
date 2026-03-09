import numpy as np
import time
from solver import solve

# Test with minimal case_spec (what evaluator might pass)
case_spec_minimal = {
    "pde": {
        "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
        "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
        "time": {
            "t_end": 0.2,
            "dt": 0.005,
            "scheme": "backward_euler",
        },
    },
}

t0 = time.time()
result = solve(case_spec_minimal)
elapsed = time.time() - t0
print(f"Minimal spec: range=[{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}], time={elapsed:.2f}s")

# Test with empty pde
case_spec_empty = {}
t0 = time.time()
result2 = solve(case_spec_empty)
elapsed = time.time() - t0
print(f"Empty spec: range=[{np.nanmin(result2['u']):.6f}, {np.nanmax(result2['u']):.6f}], time={elapsed:.2f}s")

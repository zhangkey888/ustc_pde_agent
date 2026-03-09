import time
import numpy as np
from solver import solve

# Construct case_spec matching the problem description
case_spec = {
    "pde": {
        "type": "linear_elasticity",
        "parameters": {"E": 1.0, "nu": 0.3},
        "source": ["0.0", "0.0"],
    },
    "domain": {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
    },
    "boundary_conditions": [
        {"type": "dirichlet", "location": "bottom", "value": [0.0, 0.0]},
        {"type": "dirichlet", "location": "top", "value": [0.1, 0.0]},
        {"type": "dirichlet", "location": "left", "component": 1, "value": 0.0},
        {"type": "dirichlet", "location": "right", "component": 1, "value": 0.0},
    ],
    "output": {
        "nx": 50,
        "ny": 50,
        "field": "displacement_magnitude",
    },
}

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0
u_grid = result["u"]
print(f"Shape: {u_grid.shape}")
print(f"Min: {np.nanmin(u_grid):.6e}, Max: {np.nanmax(u_grid):.6e}")
print(f"Mean: {np.nanmean(u_grid):.6e}")
print(f"Any NaN: {np.any(np.isnan(u_grid))}")
print(f"Time: {elapsed:.3f}s")
print(f"Info: {result['solver_info']}")

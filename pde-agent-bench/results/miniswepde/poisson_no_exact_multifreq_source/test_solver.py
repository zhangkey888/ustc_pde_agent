import time
import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0}
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.3f} seconds")
print("Solver info:")
for key, value in result['solver_info'].items():
    print(f"  {key}: {value}")
print(f"Solution min: {np.min(result['u']):.6f}, max: {np.max(result['u']):.6f}")
print(f"Solution shape: {result['u'].shape}")

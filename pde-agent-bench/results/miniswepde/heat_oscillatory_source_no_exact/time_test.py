import time
import solver
import numpy as np

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    },
    "coefficients": {
        "kappa": 0.8
    }
}

start = time.time()
result = solver.solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.3f} seconds")
print(f"Time limit: 48.173 seconds")
print(f"Within limit: {end - start <= 48.173}")

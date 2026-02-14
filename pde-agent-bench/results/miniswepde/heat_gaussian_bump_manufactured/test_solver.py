import time
import numpy as np

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

from solver import solve

start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.3f} seconds")
print("Solver info:", result["solver_info"])
print("Solution min/max:", np.min(result["u"]), np.max(result["u"]))

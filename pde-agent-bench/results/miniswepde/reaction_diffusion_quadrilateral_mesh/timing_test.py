import time
import numpy as np
import solver

case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "time": {
            "t_end": 0.4,
            "dt": 0.01,
            "scheme": "backward_euler"
        },
        "reaction": {
            "type": "linear"
        }
    }
}

start_time = time.time()
result = solver.solve(case_spec)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.3f} seconds")
print(f"Time limit: 114.298 seconds")
print(f"Within time limit: {end_time - start_time <= 114.298}")

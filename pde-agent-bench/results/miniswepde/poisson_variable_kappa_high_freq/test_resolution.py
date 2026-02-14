import numpy as np
from solver import solve
import time

case_spec = {
    "pde": {
        "type": "elliptic",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.3*sin(8*pi*x)*sin(8*pi*y)"}
        }
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()
print("Time:", end - start)
print("Solver info:", result["solver_info"])

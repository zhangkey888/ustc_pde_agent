import time
import numpy as np
from solver import solve

case_spec = {}  # empty dict
start = time.time()
result = solve(case_spec)
end = time.time()
print(f"Time with empty spec: {end-start:.3f}s")
print("Solver info:", result["solver_info"])
print("Solution shape:", result["u"].shape)
print("Solution min/max:", np.min(result["u"]), np.max(result["u"]))

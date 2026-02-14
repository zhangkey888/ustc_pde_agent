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
print("Time taken:", end - start)

u_computed = result["u"]
# Load reference
ref = np.load('oracle_output/reference.npz')
u_exact = ref['u_star']
# Compute error
error = np.abs(u_computed - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print("Max error:", max_error)
print("L2 error:", l2_error)
print("Accuracy requirement: <= 1.28e-03")
print("Pass?", max_error <= 1.28e-03)

import numpy as np
from solver import solve
import time

# Test case as per problem description
case_spec = {
    "pde": {
        "type": "elliptic",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.3*sin(8*pi*x)*sin(8*pi*y)"}
        }
    }
}

print("Running solve...")
start = time.time()
result = solve(case_spec)
end = time.time()
print(f"Time: {end - start:.3f}s")
print(f"Time limit: 2.569s, within limit: {end - start <= 2.569}")

u = result["u"]
print(f"u shape: {u.shape}")
print(f"u min, max: {u.min()}, {u.max()}")
print(f"u finite: {np.all(np.isfinite(u))}")

solver_info = result["solver_info"]
print("Solver info keys:", solver_info.keys())
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for key in required:
    if key not in solver_info:
        print(f"Missing key: {key}")
    else:
        print(f"{key}: {solver_info[key]}")

# Accuracy test
ref = np.load('oracle_output/reference.npz')
u_exact = ref['u_star']
error = np.abs(u - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print(f"Max error: {max_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
print(f"Accuracy requirement: <= 1.28e-3, pass: {max_error <= 1.28e-3}")

if max_error <= 1.28e-3 and end - start <= 2.569:
    print("All tests passed!")
else:
    print("Some tests failed.")

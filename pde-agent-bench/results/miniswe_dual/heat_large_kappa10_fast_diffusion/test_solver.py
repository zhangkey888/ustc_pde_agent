import time
import numpy as np

start = time.time()
from solver import solve

case_spec = {
    "pde": {
        "coefficients": {"kappa": 10.0},
        "time": {"t_end": 0.05, "dt": 0.005, "scheme": "backward_euler"}
    },
    "output": {"nx": 50, "ny": 50}
}

result = solve(case_spec)
elapsed = time.time() - start

u_grid = result["u"]
info = result["solver_info"]

# Compute error against exact solution
x_eval = np.linspace(0, 1, 50)
y_eval = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
u_exact = np.exp(-0.05) * np.sin(np.pi * X) * np.sin(np.pi * Y)

l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
linf_error = np.max(np.abs(u_grid - u_exact))

print(f"=== RESULTS ===")
print(f"L2 error:   {l2_error:.6e}  (threshold: 4.64e-04)")
print(f"Linf error: {linf_error:.6e}")
print(f"Wall time:  {elapsed:.3f}s  (threshold: 9.040s)")
print(f"Solver info: {info}")
print()

if l2_error <= 4.64e-04:
    print("ACCURACY: PASS ✅")
else:
    print("ACCURACY: FAIL ❌")

if elapsed <= 9.040:
    print("TIME: PASS ✅")
else:
    print("TIME: FAIL ❌")

# Check required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
            "iterations", "dt", "n_steps", "time_scheme"]
missing = [f for f in required if f not in info]
if missing:
    print(f"MISSING FIELDS: {missing} ❌")
else:
    print("ALL FIELDS PRESENT ✅")

# Check u_initial
if "u_initial" in result:
    print("u_initial present ✅")
else:
    print("u_initial missing (optional)")

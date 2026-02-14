import numpy as np
from solver import solve

# Simulate the evaluator call
case_spec = {
    "pde": {
        "type": "elliptic",
        "coefficients": {"kappa": 1.0}
    },
    "domain": {"type": "unit_square"},
    "exact_solution": "sin(pi*x)*sin(pi*y)"
}

result = solve(case_spec)
print("Test passed!")
print("Keys in result:", list(result.keys()))
print("u shape:", result["u"].shape)
print("solver_info keys:", list(result["solver_info"].keys()))

# Verify required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for key in required:
    assert key in result["solver_info"], f"Missing {key} in solver_info"
    print(f"{key}: {result['solver_info'][key]}")

# Verify accuracy
u_grid = result["u"]
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
l2_error = np.sqrt(np.mean((u_grid - exact)**2))
print(f"L2 error: {l2_error}")
assert l2_error < 1e-6, f"Accuracy not met: L2 error = {l2_error}"
print("Accuracy requirement satisfied.")

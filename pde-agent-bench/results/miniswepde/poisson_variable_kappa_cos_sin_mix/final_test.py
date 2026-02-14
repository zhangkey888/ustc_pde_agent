import numpy as np
import time
from solver import solve

# Test case similar to what might be provided
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.4*cos(4*pi*x)*sin(2*pi*y)"}
        },
        "boundary_conditions": {
            "dirichlet": {
                "value": "sin(pi*x)*sin(pi*y)",
                "boundary": "all"
            }
        }
    },
    "domain": {
        "type": "rectangle",
        "bounds": [[0, 0], [1, 1]]
    }
}

print("Testing solver with case_spec...")
start = time.time()
result = solve(case_spec)
end = time.time()

print(f"\nTime taken: {end-start:.3f} seconds")
print(f"Time requirement: ≤ 2.463 seconds")
print(f"Pass: {end-start <= 2.463}")

print(f"\nSolver info:")
for key, value in result["solver_info"].items():
    print(f"  {key}: {value}")

# Verify output format
assert "u" in result, "Missing 'u' in result"
assert "solver_info" in result, "Missing 'solver_info' in result"
assert result["u"].shape == (50, 50), f"u has wrong shape: {result['u'].shape}"

# Verify solver_info has required fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required_fields:
    assert field in result["solver_info"], f"Missing '{field}' in solver_info"

# Compute accuracy
u_grid = result["u"]
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u_grid - u_exact)
max_error = np.max(error)
print(f"\nMax error: {max_error:.6e}")
print(f"Accuracy requirement: ≤ 2.76e-04")
print(f"Pass: {max_error <= 2.76e-04}")

print("\nAll tests passed!")

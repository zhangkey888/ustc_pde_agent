import numpy as np
import time
import sys
sys.path.insert(0, '.')
from solver import solve

# Test case
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.5*sin(6*pi*x)"}
        }
    }
}

print("Running solver...")
start = time.time()
result = solve(case_spec)
end = time.time()

print(f"\n=== Results ===")
print(f"Solve time: {end - start:.3f} seconds")
print(f"Time limit: 2.615 seconds")
print(f"Time requirement met: {end - start <= 2.615}")

# Check output structure
assert "u" in result, "Missing 'u' in output"
assert "solver_info" in result, "Missing 'solver_info' in output"

u_grid = result["u"]
solver_info = result["solver_info"]

print(f"\nSolution array shape: {u_grid.shape}")
print(f"Expected shape: (50, 50)")
assert u_grid.shape == (50, 50), f"Wrong shape: {u_grid.shape}"

# Check solver_info fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required_fields:
    assert field in solver_info, f"Missing field in solver_info: {field}"
    print(f"{field}: {solver_info[field]}")

# Compute accuracy
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u_exact_grid = np.sin(2*np.pi*X) * np.sin(np.pi*Y)

error_grid = u_grid - u_exact_grid
l2_error = np.sqrt(np.mean(error_grid**2))
max_error = np.max(np.abs(error_grid))

print(f"\n=== Accuracy ===")
print(f"L2 error: {l2_error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Required accuracy: ≤ 7.47e-04")
print(f"Accuracy requirement met: {l2_error <= 7.47e-04}")

print("\n=== All tests passed! ===")

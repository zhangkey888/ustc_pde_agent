import numpy as np
from solver import solve

# Test case specification
case_spec = {
    "pde": {
        "type": "elliptic",
        "time": None
    }
}

# Run solver
result = solve(case_spec)
u_grid = result["u"]
solver_info = result["solver_info"]

# Create 50x50 grid for exact solution
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Compute error
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Accuracy test results:")
print(f"  Max error: {max_error:.6e}")
print(f"  Mean error: {mean_error:.6e}")
print(f"  L2 error: {l2_error:.6e}")
print(f"  Required accuracy: ≤ 5.81e-04")
print(f"  Pass: {max_error <= 5.81e-04}")

# Check solver info fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
print(f"\nSolver info check:")
for field in required_fields:
    if field in solver_info:
        print(f"  ✓ {field}: {solver_info[field]}")
    else:
        print(f"  ✗ {field}: MISSING")

# Check output shape
print(f"\nOutput shape: {u_grid.shape}")
print(f"Expected shape: (50, 50)")
print(f"Shape match: {u_grid.shape == (50, 50)}")

import time
import numpy as np
from solver import solve

print("=== Comprehensive Solver Test ===")

# Test 1: Basic case specification
print("\nTest 1: Basic Poisson case")
case_spec = {
    "pde": {
        "type": "poisson",
        "domain": {"bounds": [[0, 1], [0, 1]]}
    }
}

start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Time: {end_time - start_time:.3f}s")
print(f"Solution shape: {result['u'].shape}")
print(f"Expected shape: (50, 50)")
print(f"Shape correct: {result['u'].shape == (50, 50)}")

solver_info = result['solver_info']
print(f"\nSolver info:")
for key, value in solver_info.items():
    print(f"  {key}: {value}")

# Check required fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
missing_fields = [f for f in required_fields if f not in solver_info]
print(f"\nMissing required fields: {missing_fields}")

# Test 2: Check solution values are finite
print("\nTest 2: Solution validity")
u_grid = result['u']
print(f"Min value: {np.nanmin(u_grid):.6e}")
print(f"Max value: {np.nanmax(u_grid):.6e}")
print(f"Has NaN: {np.any(np.isnan(u_grid))}")
print(f"Has Inf: {np.any(np.isinf(u_grid))}")

# Test 3: Check against exact solution
print("\nTest 3: Accuracy check")
nx, ny = u_grid.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 5.81e-04")
print(f"Pass accuracy: {max_error <= 5.81e-04}")

# Test 4: Time requirement
print("\nTest 4: Time requirement")
print(f"Time taken: {end_time - start_time:.3f}s")
print(f"Time requirement: ≤ 2.131s")
print(f"Pass time: {(end_time - start_time) <= 2.131}")

print("\n=== All tests completed ===")

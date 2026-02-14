import solver
import numpy as np

# Test with the exact case specification from the problem
case_spec = {
    "epsilon": 0.01,
    "beta": [15.0, 0.0],
    "pde": {
        "type": "elliptic"
    }
}

print("Testing solver with case_spec:", case_spec)
result = solver.solve(case_spec)

print("\nResult keys:", list(result.keys()))
print("Solver info keys:", list(result["solver_info"].keys()))

# Verify required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required:
    if field in result["solver_info"]:
        print(f"✓ {field}: {result['solver_info'][field]}")
    else:
        print(f"✗ Missing {field}")

# Verify solution shape
u = result["u"]
print(f"\nSolution shape: {u.shape}, expected (50, 50)")
assert u.shape == (50, 50), f"Wrong shape: {u.shape}"

# Check for NaN values
nan_count = np.isnan(u).sum()
print(f"NaN values in solution: {nan_count}")
assert nan_count == 0, "Solution contains NaN values"

print("\nAll checks passed!")

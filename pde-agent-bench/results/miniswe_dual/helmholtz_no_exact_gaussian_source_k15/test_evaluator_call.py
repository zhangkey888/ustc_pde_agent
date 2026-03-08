import sys
import numpy as np
sys.path.insert(0, '.')
from solver import solve

# Minimal case spec as might be passed by evaluator
case_spec = {
    "pde": {
        "k": 15.0,
        # Note: source term not in case_spec, it's defined in problem description
    },
    "domain": {
        "bounds": [[0, 0], [1, 1]]
    }
}

print("Calling solve() with minimal case_spec...")
result = solve(case_spec)

# Verify output structure
assert "u" in result
assert "solver_info" in result

u = result["u"]
info = result["solver_info"]

assert u.shape == (50, 50)
assert isinstance(u, np.ndarray)
assert isinstance(info, dict)

required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for key in required_keys:
    assert key in info, f"Missing key: {key}"
    print(f"{key}: {info[key]}")

print(f"\nSolution stats:")
print(f"  shape: {u.shape}")
print(f"  dtype: {u.dtype}")
print(f"  min: {u.min():.6e}")
print(f"  max: {u.max():.6e}")
print(f"  mean: {u.mean():.6e}")

print("\nAll checks passed!")

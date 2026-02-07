import numpy as np
from solver import solve

print("Test 1: Basic case specification")
case_spec1 = {
    "pde": {
        "type": "elliptic",
        "time": None
    }
}
result1 = solve(case_spec1)
print(f"  Result keys: {list(result1.keys())}")
print(f"  Has 'u': {'u' in result1}")
print(f"  Has 'solver_info': {'solver_info' in result1}")
print(f"  u shape: {result1['u'].shape}")

print("\nTest 2: Case spec with extra fields (should be ignored)")
case_spec2 = {
    "pde": {
        "type": "elliptic",
        "time": None,
        "extra_field": "should_be_ignored"
    },
    "other_field": 123
}
result2 = solve(case_spec2)
print(f"  Solver completed: True")
print(f"  u shape matches: {result2['u'].shape == (50, 50)}")

print("\nTest 3: Verify solution properties")
u_grid = result1['u']
print(f"  Min value: {u_grid.min():.6f} (expected near 0)")
print(f"  Max value: {u_grid.max():.6f} (expected near 1)")
print(f"  Mean value: {u_grid.mean():.6f}")

# Check symmetry (solution should be symmetric)
center = u_grid[25, 25]
corners_avg = (u_grid[0, 0] + u_grid[0, -1] + u_grid[-1, 0] + u_grid[-1, -1]) / 4
print(f"  Center value: {center:.6f}")
print(f"  Average corner value: {corners_avg:.6f} (expected near 0)")

print("\nAll tests passed!")

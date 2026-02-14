import sys
sys.path.insert(0, '.')
from solver import solve
import numpy as np

# Minimal test case matching the problem description exactly
minimal_case = {
    'pde': {
        'coefficients': {'kappa': 1.0},
        'source': {'f': 1.0}
    },
    'domain': {
        'bounds': [[0.0, 0.0], [1.0, 1.0]]
    }
}

print("Running final minimal test...")
result = solve(minimal_case)

# Check required output structure
assert 'u' in result, "Missing 'u' in output"
assert 'solver_info' in result, "Missing 'solver_info' in output"

u = result['u']
info = result['solver_info']

# Check u shape
assert u.shape == (50, 50), f"u shape should be (50, 50), got {u.shape}"

# Check solver_info required fields
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol', 'iterations']
for field in required_fields:
    assert field in info, f"Missing required field '{field}' in solver_info"
    print(f"  {field}: {info[field]}")

# Check data types
assert isinstance(info['mesh_resolution'], int), "mesh_resolution should be int"
assert isinstance(info['element_degree'], int), "element_degree should be int"
assert isinstance(info['iterations'], int), "iterations should be int"
assert isinstance(info['rtol'], float), "rtol should be float"

print("\n✓ All checks passed!")
print(f"✓ Solution shape: {u.shape}")
print(f"✓ Solver converged at resolution: {info['mesh_resolution']}")
print(f"✓ Total linear iterations: {info['iterations']}")

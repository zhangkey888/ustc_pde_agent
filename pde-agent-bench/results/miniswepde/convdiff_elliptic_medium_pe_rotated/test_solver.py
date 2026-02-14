import numpy as np
from solver import solve

# Test 1: Basic case from problem description
print("Test 1: Basic convection-diffusion")
case_spec = {
    "pde": {
        "epsilon": 0.05,
        "beta": [3.0, 1.0]
    },
    "domain": {
        "bounds": [[0.0, 0.0], [1.0, 1.0]]
    }
}

result = solve(case_spec)
print(f"  Solution shape: {result['u'].shape}")
print(f"  Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"  Element degree: {result['solver_info']['element_degree']}")
print(f"  Time: {result['solver_info']['wall_time_sec']:.3f}s")

# Test 2: Different parameters
print("\nTest 2: Different parameters")
case_spec2 = {
    "pde": {
        "epsilon": 0.1,
        "beta": [1.0, 2.0]
    },
    "domain": {
        "bounds": [[0.0, 0.0], [1.0, 1.0]]
    }
}

result2 = solve(case_spec2)
print(f"  Solution shape: {result2['u'].shape}")
print(f"  Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"  Time: {result2['solver_info']['wall_time_sec']:.3f}s")

# Test 3: Check required fields
print("\nTest 3: Checking required fields")
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol', 'iterations']
for field in required_fields:
    if field in result['solver_info']:
        print(f"  ✓ {field}: {result['solver_info'][field]}")
    else:
        print(f"  ✗ {field} missing")

# Test 4: Check solution values are finite
print("\nTest 4: Checking solution quality")
u = result['u']
print(f"  Min: {np.min(u):.6f}")
print(f"  Max: {np.max(u):.6f}")
print(f"  Mean: {np.mean(u):.6f}")
print(f"  Has NaN: {np.any(np.isnan(u))}")
print(f"  Has Inf: {np.any(np.isinf(u))}")

print("\nAll tests completed!")

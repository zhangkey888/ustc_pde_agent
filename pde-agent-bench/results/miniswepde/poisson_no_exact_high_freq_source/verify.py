import numpy as np
from solver import solve

# Test case
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0}
    }
}

result = solve(case_spec)
print("Test passed!")
print(f"Keys in result: {list(result.keys())}")
print(f"u shape: {result['u'].shape}")
print(f"u min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")
print(f"Solver info keys: {list(result['solver_info'].keys())}")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"KSP type: {result['solver_info']['ksp_type']}")
print(f"PC type: {result['solver_info']['pc_type']}")
print(f"RTOL: {result['solver_info']['rtol']}")
print(f"Iterations: {result['solver_info']['iterations']}")

# Check that u is not all zeros or NaN
assert not np.all(result['u'] == 0), "Solution is all zeros!"
assert not np.any(np.isnan(result['u'])), "Solution contains NaN values!"
print("Solution appears valid.")

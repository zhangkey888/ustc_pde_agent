import solver
import numpy as np

# Test with various case_spec inputs
test_cases = [
    {},  # empty, should use defaults
    {"pde": {}},  # empty pde
    {"pde": {"time": {"t_end": 0.05, "dt": 0.005}}},
    {"pde": {"time": {"t_end": 0.05, "dt": 0.005, "scheme": "backward_euler"},
            "coefficients": {"kappa": 10.0}}}
]

for i, case_spec in enumerate(test_cases):
    print(f"\n=== Test case {i} ===")
    print(f"case_spec: {case_spec}")
    try:
        result = solver.solve(case_spec)
        print(f"Success: u shape {result['u'].shape}")
        print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
        print(f"Error estimate: max u = {np.max(result['u']):.6f}, min u = {np.min(result['u']):.6f}")
        # Check that u is not all zeros or NaN
        assert not np.any(np.isnan(result['u'])), "u contains NaN"
        assert np.max(np.abs(result['u'])) > 0.1, "u seems too small"
        print("✓ Basic checks passed")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n=== All tests completed ===")

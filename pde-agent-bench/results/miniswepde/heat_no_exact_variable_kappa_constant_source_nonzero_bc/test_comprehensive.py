import numpy as np
import time
from solver import solve

def test_case_variations():
    """Test various case specifications to ensure robustness."""
    
    # Base case from problem description
    base_case = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "source": 1.0,
        "initial_condition": 0.0,
        "coefficients": {
            "kappa": {
                "type": "expr",
                "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"
            }
        }
    }
    
    # Test 1: Base case
    print("Test 1: Base case (variable kappa)")
    start = time.time()
    result1 = solve(base_case)
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1:.3f}s, Mesh: {result1['solver_info']['mesh_resolution']}")
    print(f"  Iterations: {result1['solver_info']['iterations']}")
    
    # Test 2: Missing time info (should use defaults)
    print("\nTest 2: Missing time info")
    case2 = base_case.copy()
    case2["pde"] = {}  # Empty pde dict
    result2 = solve(case2)
    print(f"  dt used: {result2['solver_info']['dt']}")
    print(f"  t_end inferred: {result2['solver_info']['n_steps'] * result2['solver_info']['dt']}")
    
    # Test 3: Different dt
    print("\nTest 3: Smaller dt")
    case3 = base_case.copy()
    case3["pde"]["time"]["dt"] = 0.01
    start = time.time()
    result3 = solve(case3)
    elapsed3 = time.time() - start
    print(f"  Time: {elapsed3:.3f}s, Steps: {result3['solver_info']['n_steps']}")
    
    # Test 4: Constant kappa
    print("\nTest 4: Constant kappa")
    case4 = base_case.copy()
    case4["coefficients"]["kappa"]["expr"] = "2.0"
    result4 = solve(case4)
    print(f"  Solution max: {result4['u'].max():.6f}")
    
    # Test 5: Check output shapes and types
    print("\nTest 5: Output validation")
    print(f"  u shape: {result1['u'].shape} (expected: (50, 50))")
    print(f"  u_initial shape: {result1['u_initial'].shape} (expected: (50, 50))")
    print(f"  u dtype: {result1['u'].dtype}")
    print(f"  Has NaN: {np.any(np.isnan(result1['u']))}")
    
    # Test 6: Check solver_info completeness
    print("\nTest 6: Solver info validation")
    required_keys = [
        "mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol",
        "iterations", "dt", "n_steps", "time_scheme"
    ]
    for key in required_keys:
        if key in result1['solver_info']:
            print(f"  {key}: OK ({result1['solver_info'][key]})")
        else:
            print(f"  {key}: MISSING")
    
    # Test 7: Time constraint check
    print("\nTest 7: Performance check")
    max_time = 15.644
    all_times = [elapsed1, elapsed3]
    for i, t in enumerate(all_times, 1):
        if t > max_time:
            print(f"  Test {i}: FAIL - {t:.3f}s > {max_time}s")
        else:
            print(f"  Test {i}: PASS - {t:.3f}s <= {max_time}s")
    
    return True

if __name__ == "__main__":
    print("Running comprehensive solver tests...")
    print("=" * 60)
    test_case_variations()
    print("=" * 60)
    print("Tests completed.")

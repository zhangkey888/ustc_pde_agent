import sys
import numpy as np
from solver import solve

# Test case matching the problem description
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0}
    }
}

try:
    result = solve(case_spec)
    print("SUCCESS: Solver completed without errors")
    
    # Check required keys
    assert "u" in result, "Missing 'u' in result"
    assert "solver_info" in result, "Missing 'solver_info' in result"
    
    u = result["u"]
    info = result["solver_info"]
    
    # Check u shape
    assert u.shape == (50, 50), f"u shape is {u.shape}, expected (50, 50)"
    
    # Check solver_info required fields
    required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
    for field in required_fields:
        assert field in info, f"Missing '{field}' in solver_info"
        print(f"  {field}: {info[field]}")
    
    # Check types
    assert isinstance(info["mesh_resolution"], int), "mesh_resolution should be int"
    assert isinstance(info["element_degree"], int), "element_degree should be int"
    assert isinstance(info["iterations"], int), "iterations should be int"
    assert isinstance(info["rtol"], float), "rtol should be float"
    
    print(f"u min/max: {np.min(u):.6f}, {np.max(u):.6f}")
    print("All checks passed.")
    
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

import numpy as np
from solver import solve

def validate_result(result):
    """Validate the result structure and types."""
    errors = []
    
    # Check required keys
    required_keys = ['u', 'u_initial', 'solver_info']
    for key in required_keys:
        if key not in result:
            errors.append(f"Missing key: {key}")
    
    # Check u shape and type
    if 'u' in result:
        u = result['u']
        if not isinstance(u, np.ndarray):
            errors.append("u is not a numpy array")
        elif u.shape != (50, 50):
            errors.append(f"u has wrong shape: {u.shape}, expected (50, 50)")
    
    # Check u_initial shape and type
    if 'u_initial' in result:
        u_initial = result['u_initial']
        if not isinstance(u_initial, np.ndarray):
            errors.append("u_initial is not a numpy array")
        elif u_initial.shape != (50, 50):
            errors.append(f"u_initial has wrong shape: {u_initial.shape}, expected (50, 50)")
    
    # Check solver_info
    if 'solver_info' in result:
        info = result['solver_info']
        
        # Always required fields
        always_required = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol']
        for field in always_required:
            if field not in info:
                errors.append(f"Missing solver_info field: {field}")
        
        # Required for linear solves
        if 'iterations' not in info:
            errors.append("Missing solver_info field: iterations")
        
        # Required for time-dependent PDEs
        time_required = ['dt', 'n_steps', 'time_scheme']
        for field in time_required:
            if field not in info:
                errors.append(f"Missing solver_info field: {field}")
        
        # Check types
        if 'mesh_resolution' in info and not isinstance(info['mesh_resolution'], int):
            errors.append("mesh_resolution should be int")
        if 'iterations' in info and not isinstance(info['iterations'], int):
            errors.append("iterations should be int")
        if 'dt' in info and not isinstance(info['dt'], float):
            errors.append("dt should be float")
        if 'n_steps' in info and not isinstance(info['n_steps'], int):
            errors.append("n_steps should be int")
    
    return errors

# Test with default case
print("Testing solver with default case...")
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

result = solve(case_spec)
errors = validate_result(result)

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ All validations passed!")
    print(f"  u shape: {result['u'].shape}")
    print(f"  u_initial shape: {result['u_initial'].shape}")
    print(f"  mesh_resolution: {result['solver_info']['mesh_resolution']}")
    print(f"  solver: {result['solver_info']['ksp_type']}/{result['solver_info']['pc_type']}")
    print(f"  iterations: {result['solver_info']['iterations']}")
    print(f"  dt: {result['solver_info']['dt']}")
    print(f"  n_steps: {result['solver_info']['n_steps']}")

# Check that solution is not all zeros
if np.allclose(result['u'], 0):
    print("WARNING: u is all zeros!")
else:
    print("✓ Solution is non-zero")

# Check that initial condition is not all zeros
if np.allclose(result['u_initial'], 0):
    print("WARNING: u_initial is all zeros!")
else:
    print("✓ Initial condition is non-zero")

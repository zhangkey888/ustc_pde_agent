import numpy as np
from solver import solve

# Test with a very small dt that might cause numerical issues
test_case = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.001,  # Very small dt
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

print("Testing with very small dt (0.001)...")
print("This tests numerical stability and time-stepping robustness.")
try:
    result = solve(test_case)
    print(f"Success! Solution computed.")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Time steps: {result['solver_info']['n_steps']}")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    
    # Check that we have reasonable values
    if np.any(np.isnan(result['u'])):
        print("WARNING: Solution contains NaN values!")
    else:
        print("Solution is finite (no NaN values).")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing with large t_end to ensure time-stepping works...")
test_case2 = test_case.copy()
test_case2["pde"]["time"]["t_end"] = 0.5
test_case2["pde"]["time"]["dt"] = 0.05

try:
    result2 = solve(test_case2)
    print(f"Success! Solution computed for t_end=0.5.")
    print(f"Time steps: {result2['solver_info']['n_steps']}")
    print(f"Solution max: {result2['u'].max():.6f}")
except Exception as e:
    print(f"Error: {e}")

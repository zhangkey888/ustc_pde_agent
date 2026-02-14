import numpy as np
import time
from solver import solve

# Test case matching the problem description exactly
test_case = {
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
            "expr": "1 + 0.5*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)"
        }
    }
}

print("Running solver with exact problem specification...")
start = time.time()
result = solve(test_case)
end = time.time()

print(f"\nSolver completed in {end - start:.3f} seconds")
print(f"\nSolver info:")
for key, value in result['solver_info'].items():
    print(f"  {key}: {value}")

print(f"\nSolution statistics:")
print(f"  Shape: {result['u'].shape}")
print(f"  Min: {result['u'].min():.6f}")
print(f"  Max: {result['u'].max():.6f}")
print(f"  Mean: {result['u'].mean():.6f}")
print(f"  Std: {result['u'].std():.6f}")

print(f"\nInitial condition statistics:")
print(f"  Shape: {result['u_initial'].shape}")
print(f"  Min: {result['u_initial'].min():.6f}")
print(f"  Max: {result['u_initial'].max():.6f}")

# Check for NaN values
if np.any(np.isnan(result['u'])):
    print("WARNING: Solution contains NaN values!")
else:
    print("Solution contains no NaN values.")

# Check time constraint
if end - start > 15.644:
    print(f"WARNING: Solve time {end - start:.3f}s exceeds limit 15.644s")
else:
    print(f"Solve time {end - start:.3f}s is within limit 15.644s")

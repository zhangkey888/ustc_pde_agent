import numpy as np
import time
from solver import solve

# Test case with correct expression format (without np. prefix)
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
            "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"
        }
    }
}

print("Running solver with correct kappa expression format...")
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

# Also test with a simpler constant kappa
test_case2 = {
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
            "expr": "1.0"
        }
    }
}

print("\n\nTesting with constant kappa=1.0...")
start = time.time()
result2 = solve(test_case2)
end = time.time()
print(f"Solver completed in {end - start:.3f} seconds")
print(f"Solution max: {result2['u'].max():.6f}")

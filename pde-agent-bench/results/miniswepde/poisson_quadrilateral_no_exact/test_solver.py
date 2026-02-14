import numpy as np
import sys
sys.path.insert(0, '.')
from solver import solve

# Test case matching the problem description
test_case = {
    'pde': {
        'type': 'poisson',
        'coefficients': {'kappa': 1.0},
        'source': {'f': 1.0},
        'boundary_conditions': {
            'dirichlet': {
                'value': 0.0,  # u = 0 on boundary
                'boundary': 'all'
            }
        }
    },
    'domain': {
        'bounds': [[0.0, 0.0], [1.0, 1.0]]
    }
}

print("Testing solver with Poisson equation...")
result = solve(test_case)

u_grid = result['u']
solver_info = result['solver_info']

print(f"\nSolution statistics:")
print(f"  Shape: {u_grid.shape}")
print(f"  Min value: {u_grid.min():.6f}")
print(f"  Max value: {u_grid.max():.6f}")
print(f"  Mean value: {u_grid.mean():.6f}")
print(f"  Std dev: {u_grid.std():.6f}")

print(f"\nSolver info:")
for key, value in solver_info.items():
    print(f"  {key}: {value}")

# Check that solution is reasonable (positive in center, zero at boundaries)
center_val = u_grid[25, 25]  # Center of 50x50 grid
edge_val = u_grid[0, 0]      # Corner
print(f"\nValidation:")
print(f"  Center value (should be positive): {center_val:.6f}")
print(f"  Corner value (should be near 0): {edge_val:.6f}")

if center_val > 0 and abs(edge_val) < 0.01:
    print("\n✓ Solution looks reasonable!")
else:
    print("\n⚠ Solution may have issues!")

import numpy as np
import time
from solver import solve

# Run the solver
case_spec = {
    "pde": {
        "k": 15.0,
        "type": "helmholtz"
    },
    "domain": {
        "bounds": [[0, 0], [1, 1]]
    }
}

start = time.time()
result = solve(case_spec)
elapsed = time.time() - start

u_grid = result["u"]
solver_info = result["solver_info"]

print(f"Time: {elapsed:.3f}s (limit: 14.614s)")
print(f"Mesh: {solver_info['mesh_resolution']}, Degree: {solver_info['element_degree']}")
print(f"Solution shape: {u_grid.shape}")
print(f"Solution range: [{u_grid.min():.6e}, {u_grid.max():.6e}]")

# Compute some metrics
dx = 1.0 / 49
grid_norm = np.sqrt(np.sum(u_grid**2) * dx * dx)
print(f"Grid L2 norm: {grid_norm:.6e}")

# Check if we're using enough time
if elapsed < 0.8 * 14.614:  # Using less than 80% of time budget
    print(f"\nWARNING: Only using {elapsed/14.614*100:.1f}% of time budget.")
    print("Should increase mesh resolution or polynomial degree.")

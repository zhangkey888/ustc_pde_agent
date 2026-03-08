import time
import numpy as np
from mpi4py import MPI
from solver import solve

# Define the case specification
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.005,
            "scheme": "backward_euler"
        },
        "coefficients": {
            "kappa": 10.0
        }
    }
}

# Run solver and measure time
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

# Compute error against exact solution
if MPI.COMM_WORLD.rank == 0:
    t_end = case_spec['pde']['time']['t_end']
    u_grid = result['u']
    nx, ny = u_grid.shape
    
    # Create exact solution grid
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Compute L2 error
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    error_sq = np.sum((u_grid - u_exact)**2) * dx * dy
    l2_error = np.sqrt(error_sq)
    
    # Print results
    print(f"\n=== TEST RESULTS ===")
    print(f"Wall time: {end_time - start_time:.4f} seconds")
    print(f"L2 error: {l2_error:.4e}")
    print(f"Target error: 4.64e-04")
    print(f"Time limit: 10.406 seconds")
    print(f"\nSolver info:")
    for key, value in result['solver_info'].items():
        print(f"  {key}: {value}")
    
    # Check pass/fail
    if l2_error <= 4.64e-04 and (end_time - start_time) <= 10.406:
        print(f"\n✓ PASS: Both constraints met!")
    else:
        print(f"\n✗ FAIL: Constraints not met")
        if l2_error > 4.64e-04:
            print(f"  - Error too high: {l2_error:.4e} > 4.64e-04")
        if (end_time - start_time) > 10.406:
            print(f"  - Time too high: {end_time - start_time:.4f} > 10.406")

import numpy as np
from mpi4py import MPI
import time
import solver

# Create exact case specification as expected
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

# Run multiple times to check consistency
for run in range(3):
    print(f"\n=== Run {run+1} ===")
    start = time.time()
    result = solver.solve(case_spec)
    end = time.time()
    
    if MPI.COMM_WORLD.rank == 0:
        elapsed = end - start
        print(f"Time: {elapsed:.3f} s")
        print(f"Solver info: {result['solver_info']}")
        
        # Verify grid error
        nx, ny = 50, 50
        x_vals = np.linspace(0, 1, nx)
        y_vals = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        u_exact_grid = np.exp(-0.05) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        error_grid = np.abs(result['u'] - u_exact_grid)
        max_error = np.max(error_grid)
        l2_error = np.sqrt(np.mean(error_grid**2))
        
        print(f"Grid L2 error: {l2_error:.2e}")
        print(f"Grid max error: {max_error:.2e}")
        print(f"Target: 4.64e-04")
        
        # Check constraints
        if l2_error <= 4.64e-04 and elapsed <= 10.406:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            if l2_error > 4.64e-04:
                print(f"  Error too high: {l2_error:.2e} > 4.64e-04")
            if elapsed > 10.406:
                print(f"  Time too high: {elapsed:.3f} > 10.406")

# Also test with different dt to ensure robustness
print("\n=== Testing with smaller dt ===")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.001,  # Smaller dt
            "scheme": "backward_euler"
        },
        "coefficients": {
            "kappa": 10.0
        }
    }
}

start = time.time()
result2 = solver.solve(case_spec2)
end = time.time()

if MPI.COMM_WORLD.rank == 0:
    elapsed = end - start
    print(f"Time with dt=0.001: {elapsed:.3f} s")
    print(f"Mesh used: {result2['solver_info']['mesh_resolution']}")
    print(f"n_steps: {result2['solver_info']['n_steps']}")
    
    # Check error
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact_grid = np.exp(-0.05) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    l2_error = np.sqrt(np.mean((result2['u'] - u_exact_grid)**2))
    print(f"Grid L2 error: {l2_error:.2e}")
    
    if l2_error <= 4.64e-04 and elapsed <= 10.406:
        print("✓ PASS with smaller dt")
    else:
        print("✗ FAIL with smaller dt")

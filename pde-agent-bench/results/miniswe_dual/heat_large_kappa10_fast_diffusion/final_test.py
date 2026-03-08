import numpy as np
from mpi4py import MPI
import time
import solver

print("=== Final comprehensive test ===")

# Test 1: Default case
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

start = time.time()
result = solver.solve(case_spec)
end = time.time()

if MPI.COMM_WORLD.rank == 0:
    elapsed = end - start
    print(f"Test 1 - Default case:")
    print(f"  Time: {elapsed:.3f} s (limit: 10.406 s)")
    print(f"  Mesh: {result['solver_info']['mesh_resolution']}")
    print(f"  Degree: {result['solver_info']['element_degree']}")
    
    # Compute error
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact_grid = np.exp(-0.05) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    l2_error = np.sqrt(np.mean((result['u'] - u_exact_grid)**2))
    print(f"  Grid L2 error: {l2_error:.2e} (limit: 4.64e-04)")
    
    if l2_error <= 4.64e-04 and elapsed <= 10.406:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")

# Test 2: Different dt (should still work)
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.0025,  # Half the suggested dt
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
    print(f"\nTest 2 - Smaller dt (0.0025):")
    print(f"  Time: {elapsed:.3f} s")
    
    l2_error = np.sqrt(np.mean((result2['u'] - u_exact_grid)**2))
    print(f"  Grid L2 error: {l2_error:.2e}")
    
    if l2_error <= 4.64e-04:
        print("  ✓ Accuracy PASS")
    else:
        print("  ✗ Accuracy FAIL")

print("\n=== All tests completed ===")

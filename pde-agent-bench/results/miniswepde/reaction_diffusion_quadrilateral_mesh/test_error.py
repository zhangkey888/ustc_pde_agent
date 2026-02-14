import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
import solver

ScalarType = PETSc.ScalarType

# Create a test case
case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "time": {
            "t_end": 0.4,
            "dt": 0.01,
            "scheme": "backward_euler"
        },
        "reaction": {
            "type": "linear"
        }
    }
}

# Run solver
result = solver.solve(case_spec)

comm = MPI.COMM_WORLD
rank = comm.rank

# Compute exact solution on the same 60x60 grid
nx, ny = 60, 60
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution at t=0.4
u_exact_grid = np.exp(-0.4) * (np.exp(X) * np.sin(np.pi * Y))

# Compute error
error_grid = result['u'] - u_exact_grid
abs_error = np.max(np.abs(error_grid))
rms_error = np.sqrt(np.mean(error_grid**2))

if rank == 0:
    print(f"Maximum absolute error: {abs_error}")
    print(f"RMS error: {rms_error}")
    print(f"Required accuracy: error ≤ 8.45e-03")
    print(f"Pass accuracy test: {abs_error <= 8.45e-03}")
    
    # Check some sample points
    print("\nSample point checks:")
    idx_x, idx_y = 30, 30  # Middle of grid
    print(f"At ({x_vals[idx_x]:.2f}, {y_vals[idx_y]:.2f}):")
    print(f"  Numerical: {result['u'][idx_x, idx_y]:.6f}")
    print(f"  Exact: {u_exact_grid[idx_x, idx_y]:.6f}")
    print(f"  Error: {error_grid[idx_x, idx_y]:.6e}")

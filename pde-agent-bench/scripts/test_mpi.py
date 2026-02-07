from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"MPI rank {rank}/{size} starting test...")

# Import and run solver
from solver import solve

case_spec = {
    "pde": {
        "type": "elliptic",
        "time": None
    }
}

result = solve(case_spec)

# Only rank 0 prints results
if rank == 0:
    u_grid = result["u"]
    solver_info = result["solver_info"]
    
    print(f"\nSolver completed on {size} processes")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Min/max: {u_grid.min():.6f}, {u_grid.max():.6f}")
    print(f"Iterations: {solver_info['iterations']}")
    
    # Quick accuracy check
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"Max error: {max_error:.6e}")
    print(f"Accuracy requirement met: {max_error <= 5.81e-04}")

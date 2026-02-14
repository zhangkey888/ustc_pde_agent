import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
import time

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

# Import the solver
import solver

# Run the solver
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

start_time = time.time()
result = solver.solve(case_spec)
end_time = time.time()

if MPI.COMM_WORLD.rank == 0:
    print(f"Solve time: {end_time - start_time:.3f}s")
    print("Solver info:", result["solver_info"])
    
    # Compute error on the 50x50 grid
    u_grid = result["u"]
    nx, ny = u_grid.shape
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Exact solution at t_end = 0.1
    u_exact_grid = np.zeros_like(u_grid)
    for i in range(nx):
        for j in range(ny):
            x = np.array([X[i, j], Y[i, j], 0.0])
            u_exact_grid[i, j] = u_exact(x, 0.1)
    
    error = u_grid - u_exact_grid
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    print(f"\nError analysis:")
    print(f"  L2 error: {l2_error:.6e}")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Target error: < 2.49e-03")
    print(f"  Pass accuracy: {l2_error <= 2.49e-03}")

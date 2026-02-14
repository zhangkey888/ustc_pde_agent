import time
import numpy as np
import sys
sys.path.insert(0, '.')
from solver_fixed_v2 import solve

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Solve time: {end - start:.3f}s")
print(f"Solver info: {result['solver_info']}")

# Compute error against exact solution
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    # Exact solution at t_end
    def u_exact(x, y, t):
        return np.exp(-t) * np.exp(-40 * ((x - 0.5)**2 + (y - 0.5)**2))
    
    u_grid = result['u']
    nx, ny = u_grid.shape
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    u_exact_grid = u_exact(X, Y, 0.1)
    error = np.abs(u_grid - u_exact_grid)
    max_error = np.max(error)
    l2_error = np.sqrt(np.mean(error**2))
    
    print(f"Max error: {max_error:.6e}")
    print(f"L2 error: {l2_error:.6e}")
    print(f"Accuracy requirement: < 2.49e-03")
    print(f"Time requirement: < 13.753s")
    print(f"Pass accuracy: {l2_error <= 2.49e-03}")
    print(f"Pass time: {end - start <= 13.753}")

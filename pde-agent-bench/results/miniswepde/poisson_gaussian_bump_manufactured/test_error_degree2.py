import numpy as np
from mpi4py import MPI
import sys
sys.path.insert(0, '.')
from solver_degree2 import solve
import time

comm = MPI.COMM_WORLD
rank = comm.rank

case_spec = {"pde": {"type": "poisson"}}
start = time.perf_counter()
result = solve(case_spec)
end = time.perf_counter()
if rank == 0:
    print(f"Time taken: {end - start:.3f} seconds")
    u_grid = result["u"]
    nx, ny = u_grid.shape
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    exact = np.exp(-40 * ((X - 0.5)**2 + (Y - 0.5)**2))
    error = np.abs(u_grid - exact)
    max_err = np.max(error)
    l2_err = np.sqrt(np.mean(error**2))
    print(f"Max error: {max_err:.2e}")
    print(f"L2 error: {l2_err:.2e}")
    print("Accuracy requirement: 1.71e-03")
    print("Time requirement: 1.840s")

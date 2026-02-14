import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import solver

comm = MPI.COMM_WORLD
rank = comm.rank

# Run solver
case_spec = {"pde": {"type": "elliptic"}}
result = solver.solve(case_spec)
u_grid = result["u"]
solver_info = result["solver_info"]

# Compute exact solution on the same grid
nx = ny = 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact_grid = np.cos(np.pi * X) * np.sin(np.pi * Y)

# Compute L2 error on grid (approximate)
error_grid = u_grid - u_exact_grid
l2_error = np.sqrt(np.mean(error_grid**2))
if rank == 0:
    print(f"L2 error (grid approx): {l2_error}")
    print(f"Solver info: {solver_info}")

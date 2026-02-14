import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc

def exact_solution(x, t):
    return np.exp(-t) * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 
                         0.2 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1]))

comm = MPI.COMM_WORLD
N = 64
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Create function and interpolate exact solution at t=0.1
u_exact = fem.Function(V)
def u_exact_expr(x):
    return exact_solution(x, 0.1)
u_exact.interpolate(u_exact_expr)

# Load the solver and get solution
import solver
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
result = solver.solve(case_spec)

# We need to project the grid solution back to function space for error computation
# Instead, let's compute error at the grid points
u_grid = result['u']
nx, ny = u_grid.shape
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

# Compute exact solution at grid points
exact_vals = exact_solution([X.flatten(), Y.flatten()], 0.1)
exact_grid = exact_vals.reshape((nx, ny))

# Compute relative L2 error on grid
error_grid = u_grid - exact_grid
l2_error = np.sqrt(np.mean(error_grid**2))
max_error = np.max(np.abs(error_grid))

if comm.rank == 0:
    print(f"Grid L2 error: {l2_error:.2e}")
    print(f"Grid max error: {max_error:.2e}")
    print(f"Accuracy requirement: 4.40e-03")
    print(f"Met requirement: {l2_error <= 4.40e-03}")

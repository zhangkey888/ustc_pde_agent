import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import sys
sys.path.insert(0, '.')
from solver import solve

# Create case specification
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.5*sin(6*pi*x)"}
        }
    }
}

# Run solver
result = solve(case_spec)
u_grid = result["u"]

# Exact solution on the same grid
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u_exact_grid = np.sin(2*np.pi*X) * np.sin(np.pi*Y)

# Compute error
error_grid = u_grid - u_exact_grid
l2_error = np.sqrt(np.mean(error_grid**2))
max_error = np.max(np.abs(error_grid))

print(f"L2 error on 50x50 grid: {l2_error:.6e}")
print(f"Max error on 50x50 grid: {max_error:.6e}")
print(f"Required accuracy: ≤ 7.47e-04")
print(f"Accuracy requirement met: {l2_error <= 7.47e-04}")

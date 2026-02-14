import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

comm = MPI.COMM_WORLD
rank = comm.rank

# Simple test with small mesh
N = 16
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Exact solution function
def exact_sol(x, t):
    return np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])

# Test initial condition
u0 = fem.Function(V)
u0.interpolate(lambda x: exact_sol(x, 0.0))

# Compute L2 norm of initial condition
norm_expr = ufl.inner(u0, u0) * ufl.dx
norm_form = fem.form(norm_expr)
norm_sq = fem.assemble_scalar(norm_form)
norm = np.sqrt(norm_sq)

if rank == 0:
    print(f"Initial condition norm: {norm:.6f}")
    print(f"Expected norm (t=0): {0.5:.6f}")  # sin(2πx)*sin(πy) has norm 0.5

# Test exact solution at t=0.08
u_exact = fem.Function(V)
u_exact.interpolate(lambda x: exact_sol(x, 0.08))

# Compute error between u0 and u_exact (should not be zero)
error_expr = ufl.inner(u0 - u_exact, u0 - u_exact) * ufl.dx
error_form = fem.form(error_expr)
error_sq = fem.assemble_scalar(error_form)
error = np.sqrt(error_sq)

if rank == 0:
    print(f"Error between u(0) and u(0.08): {error:.6f}")
    print(f"exp(-0.08) = {np.exp(-0.08):.6f}")
    print(f"1 - exp(-0.08) = {1 - np.exp(-0.08):.6f}")

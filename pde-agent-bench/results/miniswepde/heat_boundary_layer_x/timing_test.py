import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

# Test a simple configuration
comm = MPI.COMM_WORLD
N = 128
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Simple Poisson problem to test timing
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType(1.0))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

start = time.time()
problem = petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="test_")
u_sol = problem.solve()
elapsed = time.time() - start

print(f"Simple solve with N={N}, P1: {elapsed:.2f}s")

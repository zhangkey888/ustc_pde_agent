import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test form
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(u, v) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

print("a_form.function_spaces:", a_form.function_spaces)
print("L_form.function_spaces:", L_form.function_spaces)
print("Type of L_form.function_spaces:", type(L_form.function_spaces))

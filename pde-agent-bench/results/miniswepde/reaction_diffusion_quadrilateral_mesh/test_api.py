from mpi4py import MPI
from dolfinx import mesh, fem
import numpy as np
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
V = fem.functionspace(domain, ("Lagrange", 1))
u = fem.Function(V)
print(f"Type of u.x: {type(u.x)}")
print(f"Has petsc_vec attribute: {hasattr(u.x, 'petsc_vec')}")
print(f"u.x properties: {dir(u.x)}")
if hasattr(u.x, 'petsc_vec'):
    print(f"u.x.petsc_vec type: {type(u.x.petsc_vec)}")

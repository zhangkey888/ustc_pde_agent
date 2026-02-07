from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
# Try tuple
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
print(V)
print(V.element)

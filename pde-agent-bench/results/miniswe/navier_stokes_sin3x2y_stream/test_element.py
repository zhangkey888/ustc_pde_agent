from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
cell = domain.ufl_cell()
# Try different methods
print("Method 1: tuple")
V1 = fem.functionspace(domain, ("Lagrange", cell, 1, (2,)))
print(V1)
print("Method 2: ufl element")
element = ufl.FiniteElement("Lagrange", cell, 1, shape=(2,))
V2 = fem.functionspace(domain, element)
print(V2)

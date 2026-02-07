from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import numpy as np
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
x = ufl.SpatialCoordinate(domain)
u_exact = ufl.as_vector([x[0], x[1]])
u_bc = fem.Function(V)
# Interpolate
u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
print(u_bc.x.array[:10])

import dolfinx.fem as fem
import dolfinx.mesh as mesh
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
V = fem.functionspace(domain, ('Lagrange', 1))
f = fem.Function(V)
# Set f to 1.0
f.x.array[:] = 1.0
# Evaluate at one point
points = np.array([[0.5, 0.5, 0.0]])
cells = np.array([0])
result = f.eval(points, cells)
print(f"Result shape: {result.shape}")
print(f"Result: {result}")

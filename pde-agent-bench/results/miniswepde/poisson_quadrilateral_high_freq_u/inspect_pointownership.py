from dolfinx import geometry
import inspect
# Create a dummy mesh to see the structure
from mpi4py import MPI
from dolfinx import mesh
import numpy as np

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
points = np.array([[0.5, 0.5, 0.0]])
data = geometry.determine_point_ownership(domain, points, padding=1e-10)

print("PointOwnershipData attributes:")
print([attr for attr in dir(data) if not attr.startswith('_')])
print("\nPointOwnershipData type:", type(data))

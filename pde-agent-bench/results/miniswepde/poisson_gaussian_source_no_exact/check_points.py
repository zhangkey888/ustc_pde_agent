import numpy as np
from mpi4py import MPI
from dolfinx import mesh, geometry

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 8, 8)
bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# Test points
points = np.array([[0.5, 0.5, 0.0], [0.1, 0.2, 0.0]])
print("Points shape:", points.shape)
print("Points:", points)

# Try both orientations
try:
    result1 = geometry.compute_collisions_points(bb_tree, points)
    print("Success with shape (2,3)")
except Exception as e:
    print("Error with shape (2,3):", e)

try:
    result2 = geometry.compute_collisions_points(bb_tree, points.T)
    print("Success with shape (3,2)")
except Exception as e:
    print("Error with shape (3,2):", e)

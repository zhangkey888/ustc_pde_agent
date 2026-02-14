from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank

# Create a simple mesh
domain = mesh.create_unit_square(comm, 4, 4)

# Create a simple function
V = fem.functionspace(domain, ("Lagrange", 1))
u = fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])  # u = x + y

# Test points
points = np.array([
    [0.25, 0.25, 0.0],
    [0.75, 0.75, 0.0],
    [0.5, 0.5, 0.0]
])

print(f"Rank {rank}: Testing with {len(points)} points")
print(f"Rank {rank}: points shape = {points.shape}")

# Determine point ownership
ownership_data = geometry.determine_point_ownership(domain, points, padding=1e-10)

print(f"Rank {rank}: dest_points shape = {ownership_data.dest_points.shape}")
print(f"Rank {rank}: dest_cells shape = {ownership_data.dest_cells.shape}")
print(f"Rank {rank}: dest_owner = {ownership_data.dest_owner}")
print(f"Rank {rank}: src_owner = {ownership_data.src_owner}")

# Evaluate function at dest_points
if len(ownership_data.dest_points) > 0:
    vals = u.eval(ownership_data.dest_points, ownership_data.dest_cells)
    print(f"Rank {rank}: vals shape = {vals.shape}")
    print(f"Rank {rank}: vals = {vals}")
    
    # Try to map back
    for i, idx in enumerate(ownership_data.dest_owner):
        print(f"Rank {rank}: dest_owner[{i}] = {idx} corresponds to point {ownership_data.dest_points[i]}")

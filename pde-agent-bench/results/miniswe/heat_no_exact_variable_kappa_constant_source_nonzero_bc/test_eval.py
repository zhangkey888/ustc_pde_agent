import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, Function
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

# Create a simple mesh
comm = MPI.COMM_WORLD
msh = mesh.create_unit_square(comm, 10, 10)
V = functionspace(msh, ("Lagrange", 1))

# Create a simple function
u = Function(V)
u.interpolate(lambda x: x[0] + x[1])

# Points to evaluate
points = np.array([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0]])

# Create bounding box tree
tree = bb_tree(msh, msh.topology.dim)

# Compute collisions
cell_candidates = compute_collisions_points(tree, points)

# Compute colliding cells
colliding_cells = compute_colliding_cells(msh, cell_candidates, points)

print("Points:", points)
print("Cell candidates:", cell_candidates.array)
print("Colliding cells:", colliding_cells.array)

# Now evaluate the function at these points
# We need to find which cells contain each point and evaluate
for i, point in enumerate(points):
    cells = colliding_cells.links(i)
    if len(cells) > 0:
        print(f"Point {point[:2]} is in cell {cells[0]}")
        # Evaluate function at point
        # This requires mapping to reference coordinates
        # For now, just get the value at the closest vertex
        pass

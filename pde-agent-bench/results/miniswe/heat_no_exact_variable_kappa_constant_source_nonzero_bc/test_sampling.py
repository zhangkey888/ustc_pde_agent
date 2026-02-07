#!/usr/bin/env python3
import numpy as np
from dolfinx import fem, mesh, default_scalar_type
from mpi4py import MPI
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

# Create a simple mesh
msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(msh, ("Lagrange", 1))

# Create a simple function
u = fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])

# Test points
n_samples = 5
x_vals = np.linspace(0.0, 1.0, n_samples)
y_vals = np.linspace(0.0, 1.0, n_samples)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(default_scalar_type)

print("Points shape:", points.shape)
print("Points dtype:", points.dtype)
print("Points flags:", points.flags)

# Create bounding box tree
tree = bb_tree(msh, msh.topology.dim)

# Try compute_collisions_points
try:
    cell_candidates = compute_collisions_points(tree, points)
    print("Success!")
except Exception as e:
    print("Error:", e)
    print("Trying with points.copy()...")
    points_copy = points.copy()
    print("Copy flags:", points_copy.flags)
    try:
        cell_candidates = compute_collisions_points(tree, points_copy)
        print("Success with copy!")
    except Exception as e2:
        print("Error with copy:", e2)

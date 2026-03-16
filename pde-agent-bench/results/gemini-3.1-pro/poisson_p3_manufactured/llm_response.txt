```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def probe_points(u_func, points_array, domain):
    """
    Evaluate a dolfinx.fem.Function at a set of points.
    u_func: dolfinx.fem.Function
    points_array: shape (3, N) numpy array
    domain: dolfinx.mesh.Mesh
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
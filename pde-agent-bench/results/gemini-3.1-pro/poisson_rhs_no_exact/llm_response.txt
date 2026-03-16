```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def probe_points(domain, u_func, points_array):
    """
    Evaluate a dolfinx function at a set of points.
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points_array.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    mesh_resolution = 64
    element_degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Boundary conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def
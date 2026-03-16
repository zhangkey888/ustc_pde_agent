```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 64, 64
    degree = 2
    t_end = 0.12
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.x.array[:] = 0.0
    
    u_initial = u_n.x.array.copy()
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
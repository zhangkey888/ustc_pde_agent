```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Spatial and temporal parameters
    nx, ny = 100, 100
    degree = 1
    kappa = 0.8
    t_end = 0.12
    dt = 0.02
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition (u0 = 0.0)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_initial = u_n.x.array.copy()
    
    # Boundary conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs =
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the Helmholtz equation:
      -∇²u - k² u = 0   in Ω
                u = 1   on ∂Ω
    
    Domain: [0, 1] x [0, 1]
    Wavenumber: k = 12.0
    """
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 128, 128
    degree = 2
    k = 12.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, f
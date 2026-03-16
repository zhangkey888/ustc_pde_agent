```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx_mesh, ny_mesh = 96, 96
    degree_u = 2
    degree_p = 1
    nu = 1.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # Define mixed function space (Taylor-Hood P2-P1)
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    TH = ufl.MixedElement([P2, P1])
    W = fem.functionspace(domain, TH)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
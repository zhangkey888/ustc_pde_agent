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
    nx, ny = 100, 100
    degree = 2
    epsilon = 0.05
    beta_val = [4.0, 0.0]
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for BC and RHS
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary Conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
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
    mesh_resolution = 64
    element_degree = 2
    epsilon = 0.05
    beta_val = [4.0, 2.0]
    
    # Create mesh
    domain = mesh.create_unit_square(
        comm, 
        nx=mesh_resolution, 
        ny=mesh_resolution, 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary conditions (u = 0 on the entire boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u
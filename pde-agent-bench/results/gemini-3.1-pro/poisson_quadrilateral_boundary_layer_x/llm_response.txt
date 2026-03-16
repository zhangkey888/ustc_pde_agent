```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Discretization and solver parameters
    mesh_resolution = 128
    element_degree = 3
    ksp_type = "cg"
    pc_type = "gamg"
    rtol = 1e-9
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(
        comm, 
        nx=mesh_resolution, 
        ny=mesh_resolution, 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.
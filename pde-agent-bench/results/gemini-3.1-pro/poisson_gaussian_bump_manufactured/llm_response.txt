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
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    # 1. Create Mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    
    # 2. Define Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x
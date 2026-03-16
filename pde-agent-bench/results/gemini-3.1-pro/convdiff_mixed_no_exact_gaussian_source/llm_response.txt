```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Function Space
    # Using a fine mesh to ensure high accuracy for the convection-dominated problem
    nx_mesh, ny_mesh = 256, 256
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # Linear elements (P1)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # 2. Boundary Conditions
    # u = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(
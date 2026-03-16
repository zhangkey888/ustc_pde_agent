```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Function Space
    nx, ny = 64, 64
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    element_degree = 1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 2. Boundary Conditions
    # Homogeneous Dirichlet BCs (since initial condition is 0 on boundaries)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 3. Initial Condition
    x = ufl.SpatialCoordinate(domain)
    u0_expr = ufl.
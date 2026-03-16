```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    epsilon = 0.2
    beta_val = [1.0, 0.5]
    
    # Mesh
    nx, ny = 64, 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
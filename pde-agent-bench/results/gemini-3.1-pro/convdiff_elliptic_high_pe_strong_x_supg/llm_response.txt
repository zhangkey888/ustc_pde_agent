```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Problem parameters
    epsilon = 0.01
    beta_val = [15.0, 0.0]
    
    # Mesh and function space
    nx, ny = 100, 100
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (u = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V
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
    nx, ny = 64, 64
    degree = 2
    k = 8.0
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for BC and RHS
    def exact_solution(x):
        return np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.full(x.shape[1], True)
        
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u
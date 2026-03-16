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
    kappa = 5.0
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for BC and source term
    # u = cos(2*pi*x) * cos(3*pi*y)
    u_exact_expr = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])
    
    # f = - div(kappa * grad(u_exact))
    #   = 5.0 * ( (2*pi)^2 + (3*pi)^2 ) * u_exact
    #   = 65.0 * pi^2 * u_exact
    f_expr
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    nx, ny = 150, 150
    degree = 2
    k = 20.0
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = sin(6*pi*x)*sin(5*pi*y)
    u_exact_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Source term: f = -Laplace(u) - k^2 * u
    # Laplace(u) = -(36*pi^2 + 25*pi^2) * u = -61*pi^2 * u
    f_expr = (61 * ufl.pi**2 - k**2) * u_exact_
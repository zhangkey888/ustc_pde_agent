```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    nx = ny = 64
    degree = 2
    t_end = 0.2
    dt = 0.02
    epsilon = 0.05
    beta_val = [2.0, 1.0]
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution expression
    u_exact = ufl.exp(-2*t) * ufl.sin(ufl.pi*x[0]) * ufl.
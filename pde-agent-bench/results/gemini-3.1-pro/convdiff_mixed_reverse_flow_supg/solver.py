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
    nx, ny = 150, 150
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # 2. Problem Parameters
    eps = 0.005
    beta_val = [-20.0, 5.0]
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # 3. Exact Solution and Source Term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # f = -eps * div(grad(u)) + beta . grad(u)
    f_expr = -eps * (1.0 - ufl.pi**2) * u_exact \
             - 20.0 * u_exact \
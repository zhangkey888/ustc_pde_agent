```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    epsilon = 0.01
    beta_val = [-12.0, 6.0]
    
    nx, ny = 128, 128
    degree = 2
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Exact Solution and Forcing Term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # f = -ε ∇²u + β·∇u
    f = -eps * ufl.div(ufl.grad(u
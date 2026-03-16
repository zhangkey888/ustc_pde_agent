```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the indefinite Helmholtz equation:
      -∇²u - k² u = f   in Ω
                u = 0   on ∂Ω
    
    Domain: [0, 1] x [0, 1]
    k = 24.0
    Exact solution: u = sin(5*pi*x)*sin(4*pi*y)
    """
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Function Space
    nx, ny = 128, 128
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # 2. Problem Parameters & Manufactured Solution
    k = 24.0
    x_ufl = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u_exact = sin(5
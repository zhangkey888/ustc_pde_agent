```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Agent-selectable parameters
    nx, ny = 64, 64
    degree = 2
    t_end = 0.25
    dt = 0.005
    epsilon = 0.1
    alpha = 1.0
    beta = -1.0
    
    comm = MPI.COMM_WORLD
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution expression
    u_ex = ufl.exp(-t) * 0.15 * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term derived from exact solution
    # f = ∂u_ex/∂t - ε
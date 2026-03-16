```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Discretization parameters
    nx, ny = 64, 64
    degree = 2
    
    # PDE parameters
    eps_val = 0.02
    beta_val = [-8.0, 4.0]
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    eps = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # Exact solution for manufactured source and boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Derive
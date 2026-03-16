```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 64, 64
    degree = 1
    epsilon = 0.1
    dt = 0.01
    t_end = 0.3
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution for manufactured source term
    u_ex = ufl.exp(-t_const) * (0.2 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # Time derivative of exact solution
    du_dt = -u_ex
    
    # Laplacian of exact solution
    lap_u = u
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
    nx, ny = 100, 100
    degree = 3
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for manufactured problem
    u_ex = ufl.as_vector((
        ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1]),
        ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    ))
    
    # Material parameters
    E = 1.0
    nu = 0.28
    mu = E / (2.0 * (1.0 + nu))
    lmbda
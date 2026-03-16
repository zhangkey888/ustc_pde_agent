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
    degree = 2
    
    # Material parameters
    E = 1.0
    nu = 0.49
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # Define exact solution for deriving source term and boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector((ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]),
                          ufl.sin(ufl.pi*x[0])*ufl.cos(
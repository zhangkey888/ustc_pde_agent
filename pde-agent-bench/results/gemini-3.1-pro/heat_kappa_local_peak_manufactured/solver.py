```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    nx, ny = 100, 100
    degree = 2
    t_end = 0.1
    dt = 0.005
    n_steps = int(round(t_end / dt))
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution and coefficients
    u_ex = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0]
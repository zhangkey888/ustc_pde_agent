```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Spatial and temporal parameters
    nx, ny = 64, 64
    degree = 2
    dt = 0.008
    t_end = 0.08
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time variable and constants
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution expression: u = exp(-t) * exp(5y) * sin(pi*x)
    u_exact_expr = ufl.exp(-t + 5.0 * x[1]) * ufl.sin
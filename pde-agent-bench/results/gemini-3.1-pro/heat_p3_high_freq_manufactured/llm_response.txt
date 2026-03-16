```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    nx, ny = 64, 64
    degree = 2
    t_end = 0.08
    dt = 0.001  # Using a smaller dt than suggested to ensure high accuracy
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    comm = MPI.COMM_WORLD
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(3*pi*x)*sin(3*pi*y)
    x = ufl.SpatialCoordinate(domain)
    u_init_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.
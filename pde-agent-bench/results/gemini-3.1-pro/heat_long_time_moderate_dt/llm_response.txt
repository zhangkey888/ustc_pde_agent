```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 64, 64
    degree = 2
    kappa = 0.5
    t_end = 0.2
    dt = 0.01
    n_steps = int(round(t_end / dt))
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Current and previous solutions
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
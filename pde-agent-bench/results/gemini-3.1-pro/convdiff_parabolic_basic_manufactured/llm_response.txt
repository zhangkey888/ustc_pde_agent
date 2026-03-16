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
    dt = 0.02
    t_end = 0.1
    epsilon = 0.1
    beta_val = [1.0, 0.5]
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    expr = fem.Expression(u_init_expr, V.element.interpolation_points)
    u_n.interpolate(expr)
    
    # Save
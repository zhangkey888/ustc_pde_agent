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
    nx, ny = 100, 100
    degree = 2
    t_end = 0.1
    dt = 0.005
    n_steps = int(round(t_end / dt))
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    
    u_exact_expr = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    u_n_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_n_expr)
    
    # Boundary condition (u =
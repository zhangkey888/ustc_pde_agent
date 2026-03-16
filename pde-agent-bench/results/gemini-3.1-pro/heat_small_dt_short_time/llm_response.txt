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
    t_end = 0.06
    dt = 0.003
    kappa = 1.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution expression for initial condition and source
    def exact_solution(t):
        return ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Interpolate initial condition
    u_initial_expr = fem.Expression(exact_solution(0.0), V.element.interpolation_points)
    u_n.
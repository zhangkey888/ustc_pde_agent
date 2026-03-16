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
    kappa = 0.5
    t_end = 0.2
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = np.exp(-2.0 * 0.0) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Interpolate initial condition
    u_init_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
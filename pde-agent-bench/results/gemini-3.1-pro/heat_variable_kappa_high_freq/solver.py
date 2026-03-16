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
    dt = 0.005
    t_end = 0.1
    n_steps = int(round(t_end / dt))
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution and source term
    u_ex = ufl.exp(-t_ufl) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    dt_u_ex = -ufl.exp(-t_ufl) * ufl.sin(2 * ufl.pi * x[0]) * ufl
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    nx, ny = 64, 64
    degree = 2
    t_end = 0.2
    dt_val = 0.005
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Time variables
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # 4. Exact solution and Source term (Manufactured Solution)
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.exp(-t) * 0.25 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Time derivative of
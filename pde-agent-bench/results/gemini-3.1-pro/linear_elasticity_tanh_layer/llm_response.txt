```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Discretization and solver parameters
    nx, ny = 80, 80
    degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-11
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # 3. Exact Solution and Material Parameters
    x = ufl.SpatialCoordinate(domain)
    u_ex_0 = ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0])
    u_ex_1 = 0.1 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u
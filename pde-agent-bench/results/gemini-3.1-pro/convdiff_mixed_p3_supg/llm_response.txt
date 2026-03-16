```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    eps_val = 0.01
    beta_val = [12.0, 4.0]
    
    mesh_res = 64
    degree = 3
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx=mesh_res, ny=mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    eps = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # 3. Exact Solution and Source Term
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Derive f exactly from the manufactured solution
    f = -eps * ufl.div(
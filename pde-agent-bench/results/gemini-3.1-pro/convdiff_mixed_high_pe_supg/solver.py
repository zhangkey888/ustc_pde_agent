```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    # Problem parameters
    epsilon = 0.005
    beta_val = [20.0, 10.0]
    
    # Discretization and solver parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for deriving the source term
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f =
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    mesh_resolution = 128
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    # 2. Mesh Generation
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Exact Solution and Coefficients
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(0.5*x)*sin(2*pi*y)
    u_exact = ufl.exp(0.5 * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    # Coefficient kappa
    kappa = 1.0 + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.25)**2 + (x[1] - 0.
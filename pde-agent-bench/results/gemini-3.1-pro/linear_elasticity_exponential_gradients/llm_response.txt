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
    mesh_resolution = 100
    element_degree = 2
    ksp_type = "cg"
    pc_type = "gamg"
    rtol = 1e-9
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree, (domain.geometry.dim,)))
    
    # 3. Material Parameters
    E = 1.0
    nu = 0.33
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # 4. Exact Solution and Source Term (Manufactured Solution)
    x = ufl.SpatialCoordinate(domain)
    u_ex =
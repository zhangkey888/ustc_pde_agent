```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Discretization parameters
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space for vector field (displacement)
    V = fem.functionspace(domain, ("Lagrange", element_degree, (domain.geometry.dim,)))
    
    # Material parameters
    E = 1.0
    nu = 0.45
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Exact solution for manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector((ufl.sin(ufl.pi*x[0])*ufl
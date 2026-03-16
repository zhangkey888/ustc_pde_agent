```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 128, 128
    nu = 0.22
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function spaces: Taylor-Hood P2-P1
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    W = V * Q
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Boundary conditions
    # Velocity BC: Swirl inflow/outflow
    def boundary_marker(x):
        return np.full(x.shape[1], True, dtype=bool)
        
    fdim = domain.topology.dim - 1
    boundary_facets =
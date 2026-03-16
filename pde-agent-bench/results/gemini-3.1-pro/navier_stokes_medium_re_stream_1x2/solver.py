```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Mesh and Function Space Setup
    comm = MPI.COMM_WORLD
    mesh_resolution = 64
    degree_u = 3
    degree_p = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Taylor-Hood mixed elements (P3/P2) for high accuracy
    P_v = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    P_p = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    W_el = ufl.MixedElement([P_v, P_p])
    W = fem.functionspace(domain, W_el)
    
    # 2. Boundary Conditions
    # Velocity Dirichlet BC on all boundaries
    def boundary_marker(x):
        return np.full(x.shape[1], True, dtype=bool)
    
    fdim = domain.topology.dim - 1
    boundary_facets =
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh Generation
    nx, ny = 96, 96
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Taylor-Hood P2/P1)
    V_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W_el = ufl.MixedElement([V_el, P_el])
    W = fem.functionspace(domain, W_el)
    
    # 3. Boundary Conditions
    bcs = []
    
    # Velocity: Dirichlet on all boundaries
    def boundary_marker(x):
        return np.full(x.shape[1], True)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
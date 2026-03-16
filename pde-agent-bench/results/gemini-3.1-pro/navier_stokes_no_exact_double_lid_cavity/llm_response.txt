```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters and Mesh
    nu = 0.18
    nx_mesh, ny_mesh = 64, 64
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Taylor-Hood P2/P1)
    V_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W_el = ufl.MixedElement([V_el, P_el])
    W = fem.functionspace(domain, W_el)
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # 3. Boundary Conditions
    bcs = []
    
    # Top boundary: u = [1, 0]
    u_top = fem.Function(V
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Function Spaces
    mesh_res = 64
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Taylor-Hood elements (P2 for velocity, P1 for pressure)
    V_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W_el = ufl.MixedElement([V_el, Q_el])
    W = fem.functionspace(domain, W_el)
    
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    nu = 0.01
    
    # 2. Exact Solution and Source Term
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_
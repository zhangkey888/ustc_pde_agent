import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh Generation
    nx, ny = 100, 100
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Taylor-Hood P2/P1)
    V_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W_el = ufl.MixedElement([V_el, P_el])
    W = fem.functionspace(domain, W_el)
    
    # 3. Exact Solution and Source Term
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector((ufl.pi * ufl.exp(2*x[0]) * ufl.cos(ufl.pi*x[1]),
                          -2 * ufl.exp(2*x[0]) * ufl.sin(ufl.pi
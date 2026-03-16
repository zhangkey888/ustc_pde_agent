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
    nu = 0.08
    mesh_res = 64
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx=mesh_res, ny=mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function spaces (Taylor-Hood)
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    TH = ufl.MixedElement([P2, P1])
    W = fem.functionspace(domain, TH)
    
    # Boundary conditions
    W0, _ = W.sub(0).collapse()
    
    u_top = fem.Function(W0)
    u_top.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[
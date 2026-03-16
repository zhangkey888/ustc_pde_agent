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
    nx, ny = 64, 64
    nu = 0.2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Define mixed function space (Taylor-Hood P2-P1)
    v_cg2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    p_cg1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.functionspace(domain, ufl.MixedElement([v_cg2, p_cg1]))
    
    V, _ = W.sub(0).collapse()
    
    # Boundary conditions
    def top_marker(x):
        return np.isclose(x[1], 1.0)
    
    def bottom_marker(x):
        return np.isclose(x[1], 0.0)
    
    fdim = domain
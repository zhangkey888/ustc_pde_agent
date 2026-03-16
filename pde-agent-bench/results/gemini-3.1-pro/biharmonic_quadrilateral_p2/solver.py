```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 128, 128
    degree = 2
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-10
    
    # Create quadrilateral mesh
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [nx, ny], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solutions and source term
    # u = sin(2*pi*x)*sin(pi*y)
    # v = -Δu = 5*pi^2 * sin(2*pi*x)*sin(pi*y)
    # f = Δ²u = -Δv = 25*pi^4 * sin(2*pi*
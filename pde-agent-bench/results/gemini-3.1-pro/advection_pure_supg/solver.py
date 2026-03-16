```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the pure advection equation: β·∇u = f
    with SUPG stabilization.
    """
    comm = MPI.COMM_WORLD
    
    # Mesh and function space parameters
    nx, ny = 100, 100
    degree = 2
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Advection velocity
    beta_val = (10.0, 4.0)
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # Identify inflow boundary: β·n < 0
    # Since β = [10, 4], inflow is at x=0 and y=0
    def inflow_marker(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    fdim = domain.topology.dim -
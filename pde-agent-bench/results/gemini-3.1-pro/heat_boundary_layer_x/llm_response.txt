```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation:
    ∂u/∂t - ∇·(κ ∇u) = f
    
    Returns a dict with the solution evaluated on a 50x50 grid,
    the initial condition on the same grid, and solver metadata.
    """
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    nx_mesh, ny_mesh = 64, 64
    degree = 2
    dt = 0.008
    t_end = 0.08
    n_steps = int(round(t_end / dt))
    kappa_val = 1.0
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Exact Solution and Source Term
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    x = u
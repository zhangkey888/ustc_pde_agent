```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Spatial discretization parameters
    nx, ny = 100, 100
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # PDE parameters
    epsilon = 0.02
    beta_val = np.array([6.0, 3.0], dtype=PETSc.ScalarType)
    beta = fem.Constant(domain, beta_val)
    
    # Time parameters
    t_end = 0.1
    dt = 0.02
    n_steps = int(np.round(t_end / dt))
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain
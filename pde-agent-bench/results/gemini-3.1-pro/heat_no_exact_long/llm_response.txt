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
    nx = 64
    ny = 64
    degree = 1
    kappa = 0.8
    t_end = 0.2
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Keep a copy of the initial condition for evaluation
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.
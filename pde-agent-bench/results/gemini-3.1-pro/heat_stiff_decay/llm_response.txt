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
    nx, ny = 64, 64
    degree = 2
    kappa = 0.5
    t_end = 0.12
    dt = 0.006
    n_steps = int(round(t_end / dt))
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    # Initial condition
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Save initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x
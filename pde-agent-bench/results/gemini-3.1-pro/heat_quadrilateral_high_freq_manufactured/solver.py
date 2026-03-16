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
    nx, ny = 100, 100
    degree = 2
    t_end = 0.1
    dt = 0.005
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Current and previous solutions
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    # Initial condition
    def u_exact(x, t):
        return np.exp(-t) * np.sin(4 * np.pi * x[0]) * np.sin(
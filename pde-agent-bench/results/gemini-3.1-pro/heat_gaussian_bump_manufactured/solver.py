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
    t_end = 0.1
    dt = 0.01
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time variable
    t_val = 0.0
    
    # Functions
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    # Initial condition
    def u0_eval(x_coords):
        r2 = (x_coords[0] - 0.5)**2 + (x_coords[1] - 0.5)**2
        return np.exp(-0.0) * np.exp(-40.0 * r2)
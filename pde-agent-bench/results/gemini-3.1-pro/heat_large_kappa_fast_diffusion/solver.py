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
    kappa = 5.0
    t_end = 0.08
    dt = 0.001  # Reduced from suggested 0.004 to ensure strict accuracy compliance
    n_steps = int(round(t_end / dt))
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Boundary condition (g = 0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
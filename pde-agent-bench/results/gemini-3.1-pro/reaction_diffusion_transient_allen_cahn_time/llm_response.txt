```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters with defaults
    dt = case_spec.get("dt", 0.02)
    epsilon = case_spec.get("epsilon", 0.01)
    reaction_lambda = case_spec.get("reaction_lambda", 1.0)
    t_end = 0.3
    
    # Mesh and function space
    nx, ny = 96, 96
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Functions for current and previous time steps
    u = fem.Function(V)
    u_n = fem.Function(V)
    
    # Exact solution for initial condition, boundary conditions, and source term
    def u_exact_eval(x_pts, t_val):
        return 0.2 * np.exp(-0.5 * t_val) * np.sin(2 * np.pi * x_pts[0]) * np
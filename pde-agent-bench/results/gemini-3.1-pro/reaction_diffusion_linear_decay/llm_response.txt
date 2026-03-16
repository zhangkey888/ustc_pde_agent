```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters or set defaults
    epsilon = case_spec.get("epsilon", 0.1)
    alpha = case_spec.get("reaction_alpha", 1.0)
    nx = case_spec.get("mesh_resolution", 64)
    ny = nx
    dt = case_spec.get("dt", 0.01)
    t_end = case_spec.get("t_end", 0.6)
    time_scheme = case_spec.get("time_scheme", "backward_euler")
    degree = case_spec.get("element_degree", 2)
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution expression
    x = ufl.SpatialCoordinate(domain)
    
    def exact_u_expr(t_val):
        return ufl.exp(-t_val) * ufl.cos(2 *
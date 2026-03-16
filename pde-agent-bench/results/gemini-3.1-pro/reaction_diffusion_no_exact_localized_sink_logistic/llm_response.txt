```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Extract parameters (with defaults for agent-selectable ones)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    dt = case_spec.get("dt", 0.01)
    time_scheme = case_spec.get("time_scheme", "backward_euler")
    reaction_rho = case_spec.get("reaction_rho", 5.0)
    epsilon = case_spec.get("epsilon", 0.01)
    t_end = case_spec.get("t_end", 0.35)
    
    # 2. Create Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Initial Condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(
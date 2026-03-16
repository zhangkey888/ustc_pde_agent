```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec or use defaults
    nx = case_spec.get("mesh_resolution", 64)
    ny = nx
    degree = 1
    epsilon = case_spec.get("epsilon", 0.01)
    reaction_lambda = case_spec.get("reaction_lambda", 100.0)
    
    t_end = 0.1
    dt = case_spec.get("dt", 0.002)
    n_steps = int(np.round(t_end / dt))
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution expression
    u_ex = ufl.exp
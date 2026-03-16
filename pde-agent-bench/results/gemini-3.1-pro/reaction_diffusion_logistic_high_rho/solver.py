```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse parameters
    pde_params = case_spec.get("pde", {})
    solver_params = case_spec.get("solver", {})
    
    epsilon = pde_params.get("epsilon", 0.01)
    rho = pde_params.get("reaction_rho", 50.0)
    
    t_end = pde_params.get("time", {}).get("t_end", 0.2)
    dt_suggested = solver_params.get("dt", 0.005)
    time_scheme = solver_params.get("time_scheme", "backward_euler")
    
    nx = solver_params.get("mesh_resolution", 64)
    ny = nx
    degree = solver_params.get("element_degree", 1)
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain
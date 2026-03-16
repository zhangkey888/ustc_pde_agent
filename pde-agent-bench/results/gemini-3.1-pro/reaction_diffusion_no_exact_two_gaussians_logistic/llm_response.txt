```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Extract parameters from case_spec with defaults
    nx = int(case_spec.get("mesh_resolution", 64))
    ny = nx
    degree = int(case_spec.get("element_degree", 1))
    epsilon = float(case_spec.get("epsilon", 0.01))
    dt = float(case_spec.get("dt", 0.01))
    t_end = float(case_spec.get("t_end", 0.4))
    time_scheme = case_spec.get("time_scheme", "backward_euler")
    newton_max_it = int(case_spec.get("newton_max_it", 20))
    ksp_type = case_spec.get("ksp_type", "gmres")
    pc_type = case_spec.get("pc_type", "ilu")
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_
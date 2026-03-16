```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    mesh_res = case_spec.get("mesh_resolution", 64)
    degree = case_spec.get("element_degree", 2)
    epsilon = case_spec.get("epsilon", 1.0)
    dt = case_spec.get("dt", 0.005)
    t_end = case_spec.get("t_end", 0.2)
    time_scheme = case_spec.get("time_scheme", "backward_euler")
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    newton_max_it = case_spec.get("newton_max_it", 20)
    pc_type = case_spec.get("pc_type", "ilu")
    ksp_type = case_spec.get("ksp_type", "gmres")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell
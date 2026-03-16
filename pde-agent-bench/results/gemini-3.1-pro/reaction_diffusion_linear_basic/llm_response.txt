```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solves the transient reaction-diffusion equation:
    ∂u/∂t - ε ∇²u = f  in Ω × (0, T]
    u = 0              on ∂Ω
    u(x,0) = u₀(x)     in Ω
    
    Manufactured solution: u = exp(-t) * sin(pi*x) * sin(pi*y)
    """
    comm = MPI.COMM_WORLD
    
    # 1. Parse parameters
    nx = case_spec.get("mesh_resolution", 64)
    ny = nx
    degree = case_spec.get("element_degree", 1)
    epsilon = case_spec.get("epsilon", 1.0)
    ksp_type = case_spec.get("ksp_type", "cg")
    pc_type = case_spec.get("pc_type", "jacobi")
    rtol = case_spec.get("rtol", 1e-8)
    dt = case_spec.get("dt", 0.01)
    t_end = case_spec.
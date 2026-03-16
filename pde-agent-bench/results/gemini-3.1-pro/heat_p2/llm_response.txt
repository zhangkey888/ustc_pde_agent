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
    mesh_resolution = 64
    element_degree = 2
    dt = 0.01
    t_end = 0.06
    n_steps = int(round(t_end / dt))
    time_scheme = "backward_euler"
    
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-8
    
    # 1. Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 2. Trial, Test, and State Functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    # Time variable for exact solution and source
    t_var = fem.Constant(domain, PETSc.ScalarType(0.
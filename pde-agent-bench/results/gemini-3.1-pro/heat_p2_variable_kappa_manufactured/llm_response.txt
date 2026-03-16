```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Time parameters
    t_end = 0.06
    dt = 0.01
    n_steps = int(round(t_end / dt))
    
    # Spatial discretization parameters
    mesh_res = 64
    degree = 2
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution and coefficients
    u_ex = ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) *
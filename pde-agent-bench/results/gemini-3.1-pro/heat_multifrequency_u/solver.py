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
    mesh_resolution = 100
    element_degree = 2
    t_end = 0.1
    dt = 0.01
    n_steps = int(round(t_end / dt))
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-8
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time variable
    t_expr = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution components
    x = ufl.SpatialCoordinate(domain)
    S = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) +
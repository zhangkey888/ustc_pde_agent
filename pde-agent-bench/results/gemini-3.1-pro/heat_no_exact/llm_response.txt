```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters or use defaults
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    degree = case_spec.get("element_degree", 1)
    dt = case_spec.get("dt", 0.02)
    t_end = 0.1
    n_steps = int(round(t_end / dt))
    
    # 1. Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 2. Boundary Conditions (Homogeneous Dirichlet)
    def boundary_marker(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                             np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[
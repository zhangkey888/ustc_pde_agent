```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 64
    nu = 0.3
    f_val = (1.0, 0.0)
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function spaces (Taylor-Hood P2-P1)
    V_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W_el = ufl.MixedElement([V_el, Q_el])
    W = fem.functionspace(domain, W_el)
    
    # Boundary conditions
    # Outflow on x=1, no-slip on x=0, y=0, y=1
    def boundary_walls(x):
        return np.logical_or(np.isclose(x[0], 0.0),
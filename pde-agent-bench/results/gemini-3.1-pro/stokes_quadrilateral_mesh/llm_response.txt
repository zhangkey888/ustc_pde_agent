```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx_mesh, ny_mesh = 100, 100
    degree_u = 2
    degree_p = 1
    
    # Create quadrilateral mesh
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [nx_mesh, ny_mesh], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Define Taylor-Hood mixed elements (Q2/Q1)
    cell_name = domain.topology.cell_name()
    V_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(domain.geometry.dim,))
    P_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    W_el = basix.ufl.mixed_element([V_el, P_el])
    W = fem.function
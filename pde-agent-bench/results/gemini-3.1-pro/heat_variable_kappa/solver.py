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
    t_end = 0.1
    n_steps = int(round(t_end / dt))
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
    u_initial = u_n.x.array.copy()
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Time variable for exact solution and source term
    t_val =
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 80
    element_degree = 2
    t_end = 0.1
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Initial condition
    u_n = fem.Function(V)
    
    def ic_func(x_pts):
        return np.exp(-120.0 * ((x_pts[0] - 0.6)**2 + (x_pts[1] - 0.4)**2))
    
    u_n.interpolate(ic_func)
    
    # Save initial condition to return
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x
```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Discretization parameters
    nx, ny = 100, 100
    degree = 2
    
    # PDE parameters
    epsilon_val = 0.005
    beta_val = [18.0, 6.0]
    
    # Create quadrilateral mesh
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [nx, ny], cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (u = 0 on the boundary since exact solution is sin(pi*x)*sin(pi*y))
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(
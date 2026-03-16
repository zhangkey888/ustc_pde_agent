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
    nx, ny = 80, 80
    degree = 2
    eps_val = 0.01
    beta_val = [14.0, 6.0]
    
    # Mesh
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [nx, ny], cell_type=mesh.CellType.quadrilateral)
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary Conditions
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_
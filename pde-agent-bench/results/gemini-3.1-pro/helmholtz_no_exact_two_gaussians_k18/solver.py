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
    nx, ny = 200, 200
    degree = 2
    k = 18.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (Homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = 12.0 * (ufl.exp(-90.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) - 
                     ufl.exp(-90.0 * ((x[0] - 0.7)**2 + (x[1] - 0.3)**2)))
    
    # Weak form for Helmholtz: -∇²u - k²u = f
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl
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
    nx, ny = 64, 64
    degree = 2
    dt = 0.001  # Using a smaller dt than suggested to ensure high accuracy
    t_end = 0.08
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Current and previous solutions
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u0_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Interpolate initial condition
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Boundary conditions (u = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Time variable for source term
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Source term f = (5*pi^2 - 1) * exp(-t) * sin(pi*x) * sin(2*
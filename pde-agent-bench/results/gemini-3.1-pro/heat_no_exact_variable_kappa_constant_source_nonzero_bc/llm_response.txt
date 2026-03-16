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
    degree = 1
    t_end = 0.1
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Coefficients and Source Term
    kappa = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Boundary Conditions (nonzero_bc -> assume 1.0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    g_val = PETSc.ScalarType(1.0)
    bc = fem.dirichletbc(g_val, dofs_bc, V)
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler Formulation
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_const * f) * v * ufl.dx
    
    a_form =
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
    nx, ny = 128, 128
    degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for BC and f
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Source term
    f = -ufl.div(ufl.grad(u_exact))
    
    # Boundary conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
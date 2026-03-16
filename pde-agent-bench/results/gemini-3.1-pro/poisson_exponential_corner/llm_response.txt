```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    mesh_resolution = 80
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # 2. Mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Boundary Conditions
    # The exact solution is u = exp(3*(x+y))*sin(pi*x)*sin(pi*y)
    # On the boundary of the unit square, sin(pi*x) or sin(pi*y) is 0, so u = 0.
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology
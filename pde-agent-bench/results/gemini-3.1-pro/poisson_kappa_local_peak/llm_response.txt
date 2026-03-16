```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the Poisson equation with a locally peaked diffusion coefficient.
    """
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    mesh_resolution = 100
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Boundary Conditions
    # The exact solution is u = sin(pi*x)*sin(2*pi*y), which is exactly 0 on all boundaries of [0,1]x[0,1].
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
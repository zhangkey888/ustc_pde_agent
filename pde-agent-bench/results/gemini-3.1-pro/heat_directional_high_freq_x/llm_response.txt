```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry, nls, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation:
    ∂u/∂t - ∇·(κ ∇u) = f
    
    using Backward Euler time stepping.
    """
    # 1. Parameters
    mesh_resolution = 100
    element_degree = 2
    dt = 0.002  # Using a slightly smaller dt than suggested for better accuracy
    t_end = 0.08
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    comm = MPI.COMM_WORLD
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Boundary Conditions (u = 0 on all boundaries)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
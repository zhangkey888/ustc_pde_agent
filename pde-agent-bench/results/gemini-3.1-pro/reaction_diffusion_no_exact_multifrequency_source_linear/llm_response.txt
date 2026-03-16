```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters with defaults
    nx, ny = 128, 128
    epsilon_val = case_spec.get("epsilon", 0.1)
    reaction_alpha_val = case_spec.get("reaction_alpha", 1.0)
    dt_val = case_spec.get("dt", 0.01)
    t_end = case_spec.get("t_end", 0.5)
    time_scheme = case_spec.get("time_scheme", "crank_nicolson")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Boundary conditions (u = 0 on boundary)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet
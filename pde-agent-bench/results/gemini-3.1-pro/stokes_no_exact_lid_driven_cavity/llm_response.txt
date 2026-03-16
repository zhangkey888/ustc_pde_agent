```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the steady incompressible Stokes flow for a lid-driven cavity.
    """
    comm = MPI.COMM_WORLD
    
    # Problem parameters
    nu = 0.2
    mesh_resolution = 64
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    
    # Define mixed function space (Taylor-Hood P2/P1)
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    TH = ufl.MixedElement([P2, P1])
    W = fem.functionspace(domain, TH)
    
    # Subspaces for boundary conditions
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    fdim
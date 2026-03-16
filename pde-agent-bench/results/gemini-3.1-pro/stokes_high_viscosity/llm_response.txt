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
    mesh_resolution = 80
    degree_u = 2
    degree_p = 1
    nu = 5.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function spaces (Taylor-Hood P2/P1)
    v_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    p_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    th_elem = ufl.MixedElement([v_elem, p_elem])
    W = fem.functionspace(domain, th_elem)
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Exact solutions for BC and RHS
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector((
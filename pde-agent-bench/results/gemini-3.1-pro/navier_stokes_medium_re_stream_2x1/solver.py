```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 64
    nu = 0.2
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Taylor-Hood P2/P1)
    v_cg2 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    p_cg1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    w_el = basix.ufl.mixed_element([v_cg2, p_cg1])
    W = fem.functionspace(domain, w_el)
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # 3. Boundary Conditions
    # Velocity: Dirichlet on all
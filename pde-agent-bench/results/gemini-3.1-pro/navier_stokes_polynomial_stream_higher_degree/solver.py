```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh Generation
    mesh_resolution = 64
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Taylor-Hood P2/P1)
    v_cg2 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    p_cg1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    W_el = basix.ufl.mixed_element([v_cg2, p_cg1])
    W = fem.functionspace(domain, W_el)
    
    # 3. Exact Solution and Source Term
    nu = 0.22
    x = ufl.SpatialCoordinate(domain)
    
    u_ex = ufl.as_vector([
        x[0]**2 * (1 - x[
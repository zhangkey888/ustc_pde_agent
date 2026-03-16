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
    mesh_res = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-8
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Boundary Conditions
    def boundary_marker(x):
        return np.full(x.shape[1], True)
        
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.
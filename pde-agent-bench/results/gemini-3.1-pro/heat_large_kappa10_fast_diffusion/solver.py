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
    mesh_resolution = 64
    element_degree = 2
    t_end = 0.05
    dt = 0.005
    kappa = 10.0
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-8
    time_scheme = "backward_euler"
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary Conditions
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_d
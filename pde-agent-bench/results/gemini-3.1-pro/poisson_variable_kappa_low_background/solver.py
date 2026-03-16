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
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary Conditions
    def boundary_marker(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                             np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(PETSc.ScalarType(
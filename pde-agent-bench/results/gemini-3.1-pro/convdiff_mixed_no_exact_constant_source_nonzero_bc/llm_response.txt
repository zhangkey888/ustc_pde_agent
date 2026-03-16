```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the Convection-Diffusion equation with SUPG stabilization.
    -ε ∇²u + β·∇u = f   in Ω
    u = g               on ∂Ω
    """
    comm = MPI.COMM_WORLD
    
    # 1. Parameters
    epsilon = 0.005
    beta_val = [12.0, 0.0]
    f_val = 1.0
    g_val = 1.0  # Non-zero boundary condition
    
    # 2. Mesh Generation
    mesh_res = 128
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 3. Function Space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Boundary Conditions
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.full_like(x[0], g_val))
    
    fdim
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
    mesh_res = 128
    degree = 2
    ksp_type = "cg"
    pc_type = "gamg"
    rtol = 1e-8
    
    # 1. Mesh Generation
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Boundary Conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 4. Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # High frequency source term: f = sin(12*pi*x)*sin(10*pi*y)
    f = ufl.sin(12 * ufl.pi * x[0]) * ufl.sin(10 * ufl.pi * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Solver Setup and Execution
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Retrieve linear
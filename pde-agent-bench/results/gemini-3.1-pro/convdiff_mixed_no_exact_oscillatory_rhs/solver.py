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
    nx, ny = 200, 200
    degree = 1
    epsilon = 0.005
    beta_val = [15.0, 7.0]
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])
    
    # Convection velocity and diffusion coefficient
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Standard Galerkin formulation
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.dot(beta, beta) + 1e-12)
    
    # Robust tau formulation for SUPG
    tau = 1.0 / ufl.sqrt((2.0 * vnorm / h)**2 + 9.0 * (4.0 * eps / h**2)**2)
    
    # Residual of strong form (for linear elements, div(grad(u)) = 0)
    L_u = ufl.dot(beta, ufl.grad(u))
    if degree > 1:
        L_u = L_u - eps * ufl.div(ufl.grad(u))
        
    # SUPG test function
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_stab = a + ufl.inner(L_u, v_supg) * ufl.dx
    L_stab = L + ufl.inner(f, v_supg) * ufl.dx
    
    # Linear problem
    problem = petsc.LinearProblem(
        a_stab, L_stab, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    kappa = 1.0
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-t) * np.exp(-40 * r2)
    return -u_val * (1 + kappa * (6400 * r2 - 160))

# Test different mesh sizes
for N in [32, 64, 128]:
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Exact solution at t=0.1
    t = 0.1
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: u_exact(x, t))
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve -∇·(κ∇u) = f (steady state at time t)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = 1.0
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    f_fe = fem.Function(V)
    f_fe.interpolate(lambda x: f_source(x, t))
    L = ufl.inner(f_fe, v) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], 
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"spatial_{N}_"
    )
    u_sol = problem.solve()
    
    # Compute error
    u_exact_fe = fem.Function(V)
    u_exact_fe.interpolate(lambda x: u_exact(x, t))
    
    error = u_sol.x.array - u_exact_fe.x.array
    l2_error = np.sqrt(np.mean(error**2))
    print(f"N={N}: L2 error = {l2_error:.6e}")
    
    # Also compute max error
    max_error = np.max(np.abs(error))
    print(f"  Max error = {max_error:.6e}")

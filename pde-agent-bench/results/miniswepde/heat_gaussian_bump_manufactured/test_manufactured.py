import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    u = u_exact(x, t)
    du_dt = -u
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    # Laplacian of exp(-40*r^2) = (-80 + 1600*r^2) * exp(-40*r^2)
    laplacian_u = u * (-80 + 1600 * r2)
    return du_dt - laplacian_u

# Test at a point
if rank == 0:
    x_test = np.array([0.5, 0.5])
    t_test = 0.0
    print(f"u_exact at center, t=0: {u_exact(x_test, t_test)}")
    print(f"f_source at center, t=0: {f_source(x_test, t_test)}")
    
    # Check PDE: du/dt - laplacian u should equal f
    u_val = u_exact(x_test, t_test)
    du_dt = -u_val
    laplacian = u_val * (-80 + 1600*0)  # r2=0 at center
    print(f"du/dt: {du_dt}, laplacian: {laplacian}, du/dt - laplacian: {du_dt - laplacian}")
    print(f"f_source: {f_source(x_test, t_test)}")
    
    # Test at another point
    x_test2 = np.array([0.6, 0.7])
    r2 = (0.1)**2 + (0.2)**2
    print(f"\nAt x=[0.6,0.7], t=0:")
    print(f"u_exact: {u_exact(x_test2, 0.0)}")
    print(f"f_source: {f_source(x_test2, 0.0)}")
    print(f"du/dt: {-u_exact(x_test2, 0.0)}")
    laplacian = u_exact(x_test2, 0.0) * (-80 + 1600*r2)
    print(f"laplacian: {laplacian}")
    print(f"du/dt - laplacian: {-u_exact(x_test2, 0.0) - laplacian}")

# Create a simple mesh and compute error
N = 16
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Project exact solution at t=0.1
t_end = 0.1
u_exact_func = fem.Function(V)
u_exact_func.interpolate(lambda x: u_exact(x, t_end))

# Solve steady-state heat equation with source at t_end to test
u = fem.Function(V)
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

f_func = fem.Function(V)
f_func.interpolate(lambda x: f_source(x, t_end))

# Weak form: -∇·(κ∇u) = f
a = ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_func, v) * ufl.dx

# Dirichlet BC from exact solution
tdim = domain.topology.dim
fdim = tdim - 1
def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ])
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: u_exact(x, t_end))
bc = fem.dirichletbc(u_bc, dofs)

problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# Compute error
error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
error_sq = fem.assemble_scalar(error_form)
error_l2 = np.sqrt(error_sq)
if rank == 0:
    print(f"\nSteady solve at t={t_end} on N={N} mesh:")
    print(f"L2 error: {error_l2}")
    
    # Also compute norm of exact solution
    norm_form = fem.form(ufl.inner(u_exact_func, u_exact_func) * ufl.dx)
    norm_sq = fem.assemble_scalar(norm_form)
    print(f"L2 norm of exact: {np.sqrt(norm_sq)}")
    print(f"Relative error: {error_l2/np.sqrt(norm_sq)}")

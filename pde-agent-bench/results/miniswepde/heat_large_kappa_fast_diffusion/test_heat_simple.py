import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

comm = MPI.COMM_WORLD
rank = comm.rank

# Test heat equation with backward Euler on small problem
N = 8
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Define boundary condition
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

# Exact solution: u = exp(-t)*sin(πx)*sin(πy)
def exact_sol(x, t):
    return np.exp(-t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

# BC function
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: exact_sol(x, 0.0))
bc = fem.dirichletbc(u_bc, dofs)

# Initial condition
u_n = fem.Function(V)
u_n.interpolate(lambda x: exact_sol(x, 0.0))

# Parameters
kappa = 1.0
dt = 0.01
t_end = 0.01  # Just one step

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

# Source term for manufactured solution
# u_exact = exp(-t)*sin(πx)*sin(πy)
# ∂u/∂t = -exp(-t)*sin(πx)*sin(πy)
# -∇·(κ∇u) = κ*2π²*exp(-t)*sin(πx)*sin(πy)
# f = ∂u/∂t - ∇·(κ∇u) = (-1 - 2κπ²)*exp(-t)*sin(πx)*sin(πy)
t_const = fem.Constant(domain, ScalarType(0.0))
f_expr = (-1.0 - 2.0*kappa*ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

# Backward Euler
a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx

# Solve one step
t_const.value = dt  # Source term at t=dt
u_bc.interpolate(lambda x: exact_sol(x, dt))  # BC at t=dt

problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="heat_")
u_sol = problem.solve()

# Exact solution at t=dt
u_exact = fem.Function(V)
u_exact.interpolate(lambda x: exact_sol(x, dt))

# Compute error
error_expr = ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx
error_form = fem.form(error_expr)
error_sq = fem.assemble_scalar(error_form)
error = np.sqrt(error_sq)

if rank == 0:
    print(f"Error after one time step: {error:.6e}")
    print(f"Should be very small (theoretical error is O(dt) for backward Euler)")
    
    # Also compute norms
    norm_sol = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)))
    norm_exact = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)))
    print(f"Solution norm: {norm_sol:.6f}, Exact norm: {norm_exact:.6f}")

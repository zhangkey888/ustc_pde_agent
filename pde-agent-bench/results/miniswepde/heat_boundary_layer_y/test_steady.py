import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 16, 16)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test Poisson equation: -∇²u = f with u = g on boundary
# Use manufactured solution: u = exp(5*y)*sin(π*x) (at t=0)
def exact_solution(x):
    return np.exp(5*x[1]) * np.sin(np.pi*x[0])

def source_term(x):
    # For u = exp(5*y)*sin(π*x), ∇²u = (-π² + 25)*exp(5*y)*sin(π*x)
    # So -∇²u = (π² - 25)*exp(5*y)*sin(π*x) = f
    pi = np.pi
    return (pi**2 - 25) * np.exp(5*x[1]) * np.sin(pi*x[0])

# Set up variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, PETSc.ScalarType(1.0)), v) * ufl.dx  # Will fix

# Actually, let's use the LinearProblem interface which handles BCs automatically
f_func = fem.Function(V)
f_func.interpolate(source_term)
L = ufl.inner(f_func, v) * ufl.dx

# Boundary conditions
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
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

u_bc = fem.Function(V)
u_bc.interpolate(exact_solution)
bc = fem.dirichletbc(u_bc, boundary_dofs)

# Solve using LinearProblem
problem = petsc.LinearProblem(a, L, bcs=[bc], 
                             petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                             petsc_options_prefix="test_")
u_sol = problem.solve()

# Compute error
u_exact_func = fem.Function(V)
u_exact_func.interpolate(exact_solution)

error = u_sol.x.array - u_exact_func.x.array
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(np.abs(error))

print(f"Steady-state test:")
print(f"L2 error: {l2_error:.2e}")
print(f"Max error: {max_error:.2e}")

# Check if error is reasonable
if l2_error < 0.1:
    print("Steady-state solver works reasonably well")
else:
    print("Steady-state solver has large error")

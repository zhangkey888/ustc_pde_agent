import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 16, 16)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test the exact problem: u = exp(-t)*exp(5*y)*sin(π*x)
def exact_solution(x, t):
    return np.exp(-t) * np.exp(5*x[1]) * np.sin(np.pi*x[0])

def source_term(x, t):
    pi = np.pi
    return np.exp(-t) * np.exp(5*x[1]) * np.sin(pi*x[0]) * (pi**2 - 26)

# Time parameters from problem
t_end = 0.08
dt = 0.008
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps  # Adjust to exactly reach t_end

# Functions
u_n = fem.Function(V)  # u at time n
u_n1 = fem.Function(V)  # u at time n+1

# Initial condition
def u0_expr(x):
    return exact_solution(x, 0.0)
u_n.interpolate(u0_expr)

# Boundary conditions (time-dependent Dirichlet)
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

# BC function (will be updated each time step)
u_bc = fem.Function(V)

# Source term function (will be updated each time step)
f = fem.Function(V)

# Variational form for backward Euler
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

κ = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(u, v) * ufl.dx + dt * κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx

# Assemble forms
a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix (without BCs since they're time-dependent)
A = petsc.assemble_matrix(a_form)
A.assemble()

# Create RHS vector
b = petsc.create_vector(L_form.function_spaces)

# Set up solver
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)

# Time stepping
current_time = 0.0
for step in range(n_steps):
    current_time += dt
    
    # Update boundary condition
    def bc_expr(x):
        return exact_solution(x, current_time)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Update source term
    def f_expr(x):
        return source_term(x, current_time)
    f.interpolate(f_expr)
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    
    # Apply lifting for non-homogeneous BC
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Apply BC to RHS
    petsc.set_bc(b, [bc])
    
    # Solve
    ksp.solve(b, u_n1.x.petsc_vec)
    u_n1.x.scatter_forward()
    
    # Check solution magnitude
    if step == 0 or step == n_steps-1:
        print(f"Step {step}, t={current_time:.3f}: u_n1 min/max = {u_n1.x.array.min():.3f}, {u_n1.x.array.max():.3f}")
    
    # Update for next step
    u_n.x.array[:] = u_n1.x.array

# Compute error
u_exact = fem.Function(V)
def exact_at_final(x):
    return exact_solution(x, t_end)
u_exact.interpolate(exact_at_final)

error = u_n1.x.array - u_exact.x.array
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(np.abs(error))

print(f"\nFull transient test with manufactured solution:")
print(f"Final time: {t_end}")
print(f"Time steps: {n_steps}, dt: {dt}")
print(f"L2 error: {l2_error:.2e}")
print(f"Max error: {max_error:.2e}")
print(f"u_n1 min/max: {u_n1.x.array.min():.3f}, {u_n1.x.array.max():.3f}")
print(f"u_exact min/max: {u_exact.x.array.min():.3f}, {u_exact.x.array.max():.3f}")

if l2_error < 0.1:
    print("Test PASSED - error is reasonable")
else:
    print("Test FAILED - error too large")

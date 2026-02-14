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

# Simple heat equation: ∂u/∂t - ∇²u = 0 with u=0 on boundary
# Initial condition: u0 = sin(πx)sin(πy)
# Exact solution: u = exp(-2π²t) sin(πx)sin(πy)

def u0_expr(x):
    return np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

def exact_solution(x, t):
    return np.exp(-2*np.pi**2*t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

# Time parameters
t_end = 0.01
dt = 0.001
n_steps = int(t_end / dt)

# Functions
u_n = fem.Function(V)  # u at time n
u_n1 = fem.Function(V)  # u at time n+1
u_n.interpolate(u0_expr)

# Boundary conditions (homogeneous Dirichlet)
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
u_bc.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_bc, boundary_dofs)

# Variational form for backward Euler
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

κ = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(u, v) * ufl.dx + dt * κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx  # f = 0

# Assemble forms
a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix (once)
A = petsc.assemble_matrix(a_form, bcs=[bc])
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
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    
    # Apply BCs
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # Solve
    ksp.solve(b, u_n1.x.petsc_vec)
    u_n1.x.scatter_forward()
    
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

print(f"Simple transient test (∂u/∂t - ∇²u = 0):")
print(f"Final time: {t_end}")
print(f"Time steps: {n_steps}, dt: {dt}")
print(f"L2 error: {l2_error:.2e}")
print(f"Max error: {max_error:.2e}")
print(f"Expected: u should decay exponentially")
print(f"u_n1 min/max: {u_n1.x.array.min():.3f}, {u_n1.x.array.max():.3f}")
print(f"u_exact min/max: {u_exact.x.array.min():.3f}, {u_exact.x.array.max():.3f}")

if l2_error < 0.01:
    print("Simple transient test PASSED")
else:
    print("Simple transient test FAILED - error too large")

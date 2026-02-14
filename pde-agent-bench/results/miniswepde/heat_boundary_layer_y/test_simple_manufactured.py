import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 16, 16)
V = fem.functionspace(domain, ("Lagrange", 1))

# Simpler manufactured solution: u = exp(-t)*sin(πx)*sin(πy)
# This has milder gradients
def exact_solution(x, t):
    return np.exp(-t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

def source_term(x, t):
    # For u = exp(-t)*sin(πx)*sin(πy)
    # ∂u/∂t = -exp(-t)*sin(πx)*sin(πy) = -u
    # ∂²u/∂x² = -π²*exp(-t)*sin(πx)*sin(πy) = -π²*u
    # ∂²u/∂y² = -π²*exp(-t)*sin(πx)*sin(πy) = -π²*u
    # ∇²u = -2π²*u
    # f = ∂u/∂t - ∇²u = -u - (-2π²*u) = (2π² - 1)*u
    pi = np.pi
    return (2*pi**2 - 1) * np.exp(-t) * np.sin(pi*x[0]) * np.sin(pi*x[1])

# Time parameters
t_end = 0.08
dt = 0.008
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

# Functions
u_n = fem.Function(V)
u_n1 = fem.Function(V)

def u0_expr(x):
    return exact_solution(x, 0.0)
u_n.interpolate(u0_expr)

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
f = fem.Function(V)

# Variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

κ = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.inner(u, v) * ufl.dx + dt * κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix
A = petsc.assemble_matrix(a_form)
A.assemble()

b = petsc.create_vector(L_form.function_spaces)

# Solver
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)

# Time stepping
current_time = 0.0
for step in range(n_steps):
    current_time += dt
    
    # Update BC
    def bc_expr(x):
        return exact_solution(x, current_time)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Update source
    def f_expr(x):
        return source_term(x, current_time)
    f.interpolate(f_expr)
    
    # Assemble and solve
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    ksp.solve(b, u_n1.x.petsc_vec)
    u_n1.x.scatter_forward()
    
    # Update
    u_n.x.array[:] = u_n1.x.array
    
    if step == 0 or step == n_steps-1:
        print(f"Step {step}, t={current_time:.3f}: u_n1 min/max = {u_n1.x.array.min():.3f}, {u_n1.x.array.max():.3f}")

# Compute error
u_exact = fem.Function(V)
def exact_at_final(x):
    return exact_solution(x, t_end)
u_exact.interpolate(exact_at_final)

error = u_n1.x.array - u_exact.x.array
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(np.abs(error))

print(f"\nSimple manufactured solution test:")
print(f"L2 error: {l2_error:.2e}")
print(f"Max error: {max_error:.2e}")
print(f"u_n1 min/max: {u_n1.x.array.min():.3f}, {u_n1.x.array.max():.3f}")
print(f"u_exact min/max: {u_exact.x.array.min():.3f}, {u_exact.x.array.max():.3f}")

if l2_error < 0.01:
    print("Test PASSED")
else:
    print("Test FAILED")

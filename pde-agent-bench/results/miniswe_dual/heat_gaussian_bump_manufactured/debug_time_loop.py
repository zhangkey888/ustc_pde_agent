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

# Simple test with small problem
N = 32
degree = 1
dt = 0.01
t_end = 0.1

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Boundary setup
u_bc = fem.Function(V)

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

# Trial/test
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
kappa = 1.0

# Time-stepping
n_steps = int(t_end / dt)

# Functions
u_n = fem.Function(V)
u_n.interpolate(lambda x: u_exact(x, 0.0))

u_sol = fem.Function(V)

# Forms
a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = u_n * v * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix
A = petsc.assemble_matrix(a_form, bcs=[])
A.assemble()

# RHS
b = petsc.create_vector(L_form.function_spaces)

# Solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

# Source term
f_fe = fem.Function(V)

# Time loop with proper update
t = 0.0
for step in range(n_steps):
    t_new = t + dt
    
    # Update BC
    u_bc.interpolate(lambda x: u_exact(x, t_new))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Update source
    f_fe.interpolate(lambda x: f_source(x, t_new))
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)  # u_n * v
    
    # Add source
    source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
    b_source = petsc.create_vector(source_form.function_spaces)
    petsc.assemble_vector(b_source, source_form)
    b.axpy(1.0, b_source)
    
    # Apply BCs
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # Solve
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    # DEBUG: Check solution
    u_exact_fe = fem.Function(V)
    u_exact_fe.interpolate(lambda x: u_exact(x, t_new))
    error = u_sol.x.array - u_exact_fe.x.array
    l2_error = np.sqrt(np.mean(error**2))
    print(f"Step {step}, t={t_new:.3f}, L2 error: {l2_error:.3e}")
    
    # Update u_n for next step - PROPERLY
    u_n.x.array[:] = u_sol.x.array
    u_n.x.scatter_forward()  # Important!
    
    t = t_new

print(f"\nFinal error: {l2_error:.6e}")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
rank = comm.rank

# Parameters
t_end = 0.1
dt = 0.01
kappa = 1.0
n_steps = int(t_end / dt)

# Manufactured solution
def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-t) * np.exp(-40 * r2)
    return -u_val * (1 + kappa * (6400 * r2 - 160))

# Create mesh
domain = mesh.create_unit_square(comm, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary condition setup
u_bc = fem.Function(V)
def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    ])
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# Initial condition
u_n = fem.Function(V)
u_n.interpolate(lambda x: u_exact(x, 0.0))

# Trial/test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Backward Euler forms
a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
L_base = u_n * v * ufl.dx

a_form = fem.form(a)
L_base_form = fem.form(L_base)

# Assemble matrix with BCs (use dummy BC for assembly)
bc_dummy = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
A = petsc.assemble_matrix(a_form, bcs=[bc_dummy])
A.assemble()

# Create RHS vector
b = petsc.create_vector(L_base_form.function_spaces)

# Solver
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Time-stepping
u_sol = fem.Function(V)
t = 0.0

for step in range(n_steps):
    t_new = t + dt
    
    # Update BC with exact solution at t_new
    u_bc.interpolate(lambda x: u_exact(x, t_new))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Update source term
    f_fe = fem.Function(V)
    f_fe.interpolate(lambda x: f_source(x, t_new))
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_base_form)
    
    # Add source term
    L_source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
    b_source = petsc.create_vector(L_source_form.function_spaces)
    with b_source.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b_source, L_source_form)
    b.axpy(1.0, b_source)
    
    # Apply BCs
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # Solve
    ksp.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    # Update for next step
    u_n.x.array[:] = u_sol.x.array
    t = t_new

# Compute error
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(lambda x: u_exact(x, t_end))

error_func = fem.Function(V)
error_func.x.array[:] = u_sol.x.array - u_exact_fe.x.array
error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
error_l2 = np.sqrt(fem.assemble_scalar(error_form))

if rank == 0:
    print(f"Error: {error_l2:.6e}")
    
# Also test with LinearProblem for comparison
print("\nTesting with LinearProblem for comparison:")
u_n2 = fem.Function(V)
u_n2.interpolate(lambda x: u_exact(x, 0.0))
u_sol2 = fem.Function(V)
t2 = 0.0

for step in range(n_steps):
    t_new = t2 + dt
    
    u_bc2 = fem.Function(V)
    u_bc2.interpolate(lambda x: u_exact(x, t_new))
    bc2 = fem.dirichletbc(u_bc2, dofs)
    
    f_fe2 = fem.Function(V)
    f_fe2.interpolate(lambda x: f_source(x, t_new))
    
    L2 = u_n2 * v * ufl.dx + dt * ufl.inner(f_fe2, v) * ufl.dx
    
    problem = petsc.LinearProblem(a, L2, bcs=[bc2], 
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                  petsc_options_prefix="test2_")
    u_sol2 = problem.solve()
    
    u_n2.x.array[:] = u_sol2.x.array
    t2 = t_new

error_func2 = fem.Function(V)
error_func2.x.array[:] = u_sol2.x.array - u_exact_fe.x.array
error_form2 = fem.form(ufl.inner(error_func2, error_func2) * ufl.dx)
error_l2_2 = np.sqrt(fem.assemble_scalar(error_form2))

if rank == 0:
    print(f"LinearProblem error: {error_l2_2:.6e}")
    print(f"Difference between methods: {np.max(np.abs(u_sol.x.array - u_sol2.x.array))}")

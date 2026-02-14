import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

# Simple test: solve du/dt = 0 with u=exact on boundary
# Should maintain initial condition

# Create mesh
domain = mesh.create_unit_square(comm, 16, 16, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Exact solution at any t (since du/dt=0)
def u_exact(x, t):
    return np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

# Boundary condition
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

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Time-stepping parameters
t_end = 0.1
dt = 0.01
n_steps = int(t_end / dt)

# Functions
u_n = fem.Function(V)  # previous step
u_n.interpolate(lambda x: u_exact(x, 0.0))

u_sol = fem.Function(V)  # current step

# Backward Euler for du/dt = 0: (u - u_n)/dt = 0 => u = u_n
# But with BCs, should maintain exact solution
a = u * v * ufl.dx
L = u_n * v * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix
A = petsc.assemble_matrix(a_form, bcs=[])
A.assemble()

# Create RHS vector
b = petsc.create_vector(L_form.function_spaces)

# Solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

# Time-stepping
t = 0.0
for step in range(n_steps):
    t += dt
    
    # Update BC
    u_bc.interpolate(lambda x: u_exact(x, t))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    
    # Apply BCs
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # Solve
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    # Update for next step
    u_n.x.array[:] = u_sol.x.array

# Check error
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(lambda x: u_exact(x, t_end))

error = u_sol.x.array - u_exact_fe.x.array
l2_error = np.sqrt(np.mean(error**2))
print(f"Transient test L2 error (du/dt=0): {l2_error:.6e}")
print(f"Expected: close to 0")
print(f"Solution min: {u_sol.x.array.min():.6f}, max: {u_sol.x.array.max():.6f}")
print(f"Exact min: {u_exact_fe.x.array.min():.6f}, max: {u_exact_fe.x.array.max():.6f}")

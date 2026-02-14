import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 5)**2 + (x[1] - 5)**2))

# Move Gaussian to center of [0,10]^2 to avoid boundary issues
# Actually, let's use a simpler test: u = exp(-t) on domain with BC u=exp(-t)
# This is trivial but tests time integration

N = 32
degree = 1
dt = 0.01
t_end = 0.1

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# BC: u = exp(-t) on boundary
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
u_n.interpolate(lambda x: np.ones_like(x[0]))  # u(0)=1

u_sol = fem.Function(V)

# Backward Euler for du/dt = -u (no diffusion, f=0)
# (u - u_n)/dt = -u  => u = u_n/(1+dt)
# With diffusion and BCs, more complex

# Actually, solve du/dt - ∇²u = 0 with u=exp(-t) on boundary
# Exact solution: u = exp(-t) everywhere (if initial condition matches)

a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = u_n * v * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

A = petsc.assemble_matrix(a_form, bcs=[])
A.assemble()

b = petsc.create_vector(L_form.function_spaces)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

# Time loop
t = 0.0
for step in range(n_steps):
    t_new = t + dt
    
    # BC: u = exp(-t)
    u_bc.interpolate(lambda x: np.exp(-t_new) * np.ones_like(x[0]))
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
    
    # Update
    u_n.x.array[:] = u_sol.x.array
    u_n.x.scatter_forward()
    t = t_new

# Exact solution at final time: exp(-t)
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(lambda x: np.exp(-t_end) * np.ones_like(x[0]))

error = u_sol.x.array - u_exact_fe.x.array
l2_error = np.sqrt(np.mean(error**2))
print(f"Test with u=exp(-t): L2 error = {l2_error:.6e}")
print(f"Expected: small (should be exact up to BC imposition)")

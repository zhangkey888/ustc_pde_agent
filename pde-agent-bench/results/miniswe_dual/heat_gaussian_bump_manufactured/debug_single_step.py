import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Single time step test
N = 32
degree = 1
dt = 0.01
t_end = dt  # Just one step

# Create mesh
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Manufactured solution
def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-t) * np.exp(-40 * r2)
    return -u_val * (1 + 1.0 * (6400 * r2 - 160))

# Boundary condition
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
u_sol = fem.Function(V)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (u * v + dt * 1.0 * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = u_n * v * ufl.dx

# Source term at t=dt
f_fe = fem.Function(V)
f_fe.interpolate(lambda x: f_source(x, dt))

# Assemble
a_form = fem.form(a)
L_form = fem.form(L)

# Direct solver
A = petsc.assemble_matrix(a_form, bcs=[])
A.assemble()
b = petsc.create_vector(L_form.function_spaces)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")
solver.setTolerances(rtol=1e-12)
solver.setFromOptions()

# Update BC at t=dt
u_bc.interpolate(lambda x: u_exact(x, dt))
bc = fem.dirichletbc(u_bc, dofs)

# Assemble RHS
with b.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b, L_form)

# Add source
source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
b_source = petsc.create_vector(source_form.function_spaces)
petsc.assemble_vector(b_source, source_form)
b.axpy(1.0, b_source)

# Apply BC
petsc.apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

# Solve
solver.solve(b, u_sol.x.petsc_vec)
u_sol.x.scatter_forward()

# Compute error
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(lambda x: u_exact(x, dt))

error_func = fem.Function(V)
error_func.x.array[:] = u_sol.x.array - u_exact_fe.x.array

error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
error_l2 = np.sqrt(fem.assemble_scalar(error_form))

if rank == 0:
    print(f"Single step dt={dt}:")
    print(f"  FE L2 error: {error_l2:.6e}")
    
    # Check some values
    print(f"\nChecking at center point:")
    # Evaluate at center
    points = np.array([[0.5], [0.5], [0.0]])
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            val = u_sol.eval(points.T[i], np.array([links[0]], dtype=np.int32))
            val_exact = u_exact_fe.eval(points.T[i], np.array([links[0]], dtype=np.int32))
            print(f"  Numerical: {val[0]:.6e}")
            print(f"  Exact:     {val_exact[0]:.6e}")
            print(f"  Error:     {abs(val[0] - val_exact[0]):.6e}")
            break

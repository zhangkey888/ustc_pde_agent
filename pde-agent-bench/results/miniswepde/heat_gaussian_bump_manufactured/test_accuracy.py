import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

comm = MPI.COMM_WORLD
rank = comm.rank

# Use the same parameters as in solver
t_end = 0.1
dt = 0.01
n_steps = 10
N = 32
element_degree = 2

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", element_degree))

# Boundary condition dofs
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

# Time-stepping (same as solver)
def f_source(x, t):
    u = u_exact(x, t)
    du_dt = -u
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    laplacian_u = u * (-160 + 6400 * r2)
    return du_dt - laplacian_u

u_n = fem.Function(V)
u = fem.Function(V)
f_func = fem.Function(V)

u_n.interpolate(lambda x: u_exact(x, 0.0))
u.x.array[:] = u_n.x.array

v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

u_bc_dummy = fem.Function(V)
u_bc_dummy.interpolate(lambda x: np.zeros_like(x[0]))
bc_dummy = fem.dirichletbc(u_bc_dummy, dofs)

A = petsc.assemble_matrix(a_form, bcs=[bc_dummy])
A.assemble()

b = petsc.create_vector(L_form.function_spaces)
u_sol_vec = petsc.create_vector(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

t = 0.0
for step in range(n_steps):
    t += dt
    f_func.interpolate(lambda x: f_source(x, t))
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: u_exact(x, t))
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    solver.solve(b, u_sol_vec)
    u.x.array[:] = u_sol_vec.array
    u.x.scatter_forward()
    u_n.x.array[:] = u.x.array

# Sample on 50x50 grid
nx = ny = 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T

u_grid = np.zeros((nx, ny))
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

points_on_proc = []
cells_on_proc = []
eval_map = []

for i in range(points.shape[0]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        points_on_proc.append(points[i])
        cells_on_proc.append(links[0])
        eval_map.append(i)

if len(points_on_proc) > 0:
    vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
    flat_vals = vals.flatten()
    linear_indices = np.unravel_index(eval_map, (nx, ny))
    for idx, val in zip(zip(*linear_indices), flat_vals):
        u_grid[idx] = val

if comm.size > 1:
    u_grid_full = np.zeros_like(u_grid)
    comm.Reduce(u_grid, u_grid_full, op=MPI.SUM, root=0)
    if rank == 0:
        u_grid = u_grid_full
    else:
        u_grid = np.zeros((nx, ny))

if rank == 0:
    # Compute exact on grid
    exact_grid = u_exact(np.array([X, Y]), t_end)
    error_grid = np.abs(u_grid - exact_grid)
    max_error = np.max(error_grid)
    rms_error = np.sqrt(np.mean(error_grid**2))
    print(f"Max error on 50x50 grid: {max_error:.6e}")
    print(f"RMS error on grid: {rms_error:.6e}")
    print(f"Required accuracy: ≤ 2.49e-03")
    if max_error <= 2.49e-03:
        print("PASS: Accuracy requirement met.")
    else:
        print("FAIL: Accuracy requirement not met.")
    # Also compute L2 error over domain for reference
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: u_exact(x, t_end))
    error_form = fem.form(ufl.inner(u - u_exact_func, u - u_exact_func) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(error_sq)
    print(f"L2 error over domain: {error_l2:.6e}")

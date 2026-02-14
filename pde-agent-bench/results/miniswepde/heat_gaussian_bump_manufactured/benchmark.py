import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    u = u_exact(x, t)
    du_dt = -u
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    laplacian_u = u * (-160 + 6400 * r2)
    return du_dt - laplacian_u

N = 128
element_degree = 2
t_end = 0.1
dt = 0.01
n_steps = 10

if rank == 0:
    print(f"Benchmark N={N}, degree={element_degree}, dt={dt}")

start = time.time()
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", element_degree))

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

u_bc_dummy = fem.Function(V)
u_bc_dummy.interpolate(lambda x: np.zeros_like(x[0]))
bc_dummy = fem.dirichletbc(u_bc_dummy, dofs)

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

end = time.time()
if rank == 0:
    print(f"Time: {end-start:.3f}s")
    # Quick error check
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: u_exact(x, t_end))
    error_form = fem.form(ufl.inner(u - u_exact_func, u - u_exact_func) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(error_sq)
    print(f"L2 error: {error_l2:.2e}")

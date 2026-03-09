import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
N = 8
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
gdim = domain.geometry.dim
cell_name = domain.topology.cell_name()

vel_el = basix.ufl.element("Lagrange", cell_name, 2, shape=(gdim,))
pres_el = basix.ufl.element("Lagrange", cell_name, 1)
mel = basix.ufl.mixed_element([vel_el, pres_el])

W = fem.functionspace(domain, mel)
V = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, 2, shape=(gdim,)))
Q = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, 1))

# Check that u_exact is divergence-free
x = ufl.SpatialCoordinate(domain)
pi_val = ufl.pi
u_exact = ufl.as_vector([
    0.5 * pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
    -0.5 * pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
])

# div(u_exact) = 0.5*pi * pi*cos(pi*y)*cos(pi*x) + (-0.5*pi)*(-pi*sin(pi*x)*(-sin(pi*y)))
# = 0.5*pi^2*cos(pi*x)*cos(pi*y) - 0.5*pi^2*cos(pi*x)*cos(pi*y) = 0 ✓

div_form = fem.form(ufl.div(u_exact)**2 * ufl.dx)
div_val = fem.assemble_scalar(div_form)
print(f"||div(u_exact)||^2 = {div_val:.2e}")

# Now test: initialize w with exact solution and check residual
w = fem.Function(W)
(u_sol, p_sol) = ufl.split(w)
(v_test, q_test) = ufl.TestFunctions(W)

nu_val = 2.0
p_exact = ufl.cos(pi_val * x[0]) + ufl.cos(pi_val * x[1])
f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)

nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

F_form = (
    nu * ufl.inner(ufl.grad(u_sol), ufl.grad(v_test)) * ufl.dx
    + ufl.inner(ufl.grad(u_sol) * u_sol, v_test) * ufl.dx
    - p_sol * ufl.div(v_test) * ufl.dx
    + ufl.div(u_sol) * q_test * ufl.dx
    - ufl.inner(f, v_test) * ufl.dx
)

# Interpolate exact solution into w
w.sub(0).interpolate(lambda x: np.vstack([
    0.5 * np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
    -0.5 * np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
]))
w.sub(1).interpolate(lambda x: np.cos(np.pi * x[0]) + np.cos(np.pi * x[1]))
w.x.scatter_forward()

F_compiled = fem.form(F_form)
b = petsc.assemble_vector(F_compiled)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
print(f"Residual with exact solution (no BCs): {b.norm():.6e}")

# Now check with BCs
boundary_facets = mesh.locate_entities_boundary(
    domain, domain.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool)
)
u_bc_func = fem.Function(V)
u_bc_func.interpolate(lambda x: np.vstack([
    0.5 * np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
    -0.5 * np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
]))
dofs_u = fem.locate_dofs_topological((W.sub(0), V), domain.topology.dim - 1, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
bcs = [bc_u]

# Check what set_bc does
b2 = petsc.assemble_vector(F_compiled)
b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b2, bcs)
print(f"Residual with exact solution (with set_bc): {b2.norm():.6e}")

# Check with lifting
dw_trial = ufl.TrialFunction(W)
J_form = ufl.derivative(F_form, w, dw_trial)
J_compiled = fem.form(J_form)

b3 = petsc.assemble_vector(F_compiled)
petsc.apply_lifting(b3, [J_compiled], bcs=[bcs])
b3.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b3, bcs)
print(f"Residual with exact solution (with lifting+set_bc): {b3.norm():.6e}")

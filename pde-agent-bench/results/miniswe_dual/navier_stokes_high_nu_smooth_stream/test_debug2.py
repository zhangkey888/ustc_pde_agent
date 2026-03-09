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

x = ufl.SpatialCoordinate(domain)
pi_val = ufl.pi
u_exact = ufl.as_vector([
    0.5 * pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
    -0.5 * pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
])
p_exact = ufl.cos(pi_val * x[0]) + ufl.cos(pi_val * x[1])

w = fem.Function(W)
(u_sol, p_sol) = ufl.split(w)
(v_test, q_test) = ufl.TestFunctions(W)
dw_trial = ufl.TrialFunction(W)

nu_val = 2.0
f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

F_form = (
    nu * ufl.inner(ufl.grad(u_sol), ufl.grad(v_test)) * ufl.dx
    + ufl.inner(ufl.grad(u_sol) * u_sol, v_test) * ufl.dx
    - p_sol * ufl.div(v_test) * ufl.dx
    + ufl.div(u_sol) * q_test * ufl.dx
    - ufl.inner(f, v_test) * ufl.dx
)
J_form = ufl.derivative(F_form, w, dw_trial)

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

# Pin pressure
p_bc_val = fem.Function(Q)
p_bc_val.interpolate(lambda x: np.cos(np.pi * x[0]) + np.cos(np.pi * x[1]))
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
bc_p = fem.dirichletbc(p_bc_val, p_dofs, W.sub(1))

bcs = [bc_u, bc_p]

# Initialize with exact solution
w.sub(0).interpolate(lambda x: np.vstack([
    0.5 * np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
    -0.5 * np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
]))
w.sub(1).interpolate(lambda x: np.cos(np.pi * x[0]) + np.cos(np.pi * x[1]))
w.x.scatter_forward()

F_compiled = fem.form(F_form)
J_compiled = fem.form(J_form)

# Test different BC application approaches
# Approach 1: No BC application
b1 = petsc.assemble_vector(F_compiled)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
print(f"1. Raw residual: {b1.norm():.6e}")

# Approach 2: set_bc with x0=w (should give 0 at BC dofs since w satisfies BCs)
b2 = petsc.assemble_vector(F_compiled)
b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b2, bcs, w.x.petsc_vec)
print(f"2. set_bc(b, bcs, w): {b2.norm():.6e}")

# Approach 3: lifting + set_bc with x0
b3 = petsc.assemble_vector(F_compiled)
petsc.apply_lifting(b3, [J_compiled], bcs=[bcs], x0=[w.x.petsc_vec], alpha=-1.0)
b3.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b3, bcs, w.x.petsc_vec)
print(f"3. lifting(x0=w, alpha=-1) + set_bc(x0=w): {b3.norm():.6e}")

# Approach 4: Just zero out BC dofs manually
b4 = petsc.assemble_vector(F_compiled)
b4.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Zero out BC dofs
bc_dof_indices = []
for bc in bcs:
    bc_dof_indices.extend(bc._cpp_object.dof_indices()[0])
b4_arr = b4.getArray()
b4_arr[bc_dof_indices] = 0.0
b4.setArray(b4_arr)
print(f"4. Manual zero at BC dofs: {b4.norm():.6e}")

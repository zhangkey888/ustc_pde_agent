import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary condition
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
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_bc, dofs)

# Time-stepping setup
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.zeros_like(x[0]))
u = fem.Function(V)

dt = 0.03
dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
f = fem.Constant(domain, PETSc.ScalarType(1.0))

v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)

# Backward Euler
a = ufl.inner(u_trial/dt_constant, v) * ufl.dx + ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n/dt_constant + f, v) * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

print("Assembling matrix...")
A = petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()

print("Creating vector...")
b = petsc.create_vector(L_form.function_spaces)

print("Setting up solver...")
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-8, max_it=1000)

# One time step
print("Assembling RHS...")
with b.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b, L_form)
petsc.apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

print("Solving...")
solver.solve(b, u.x.petsc_vec)
u.x.scatter_forward()

print("Success! Norm:", np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx))))

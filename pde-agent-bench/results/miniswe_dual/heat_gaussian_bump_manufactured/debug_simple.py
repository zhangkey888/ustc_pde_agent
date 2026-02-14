import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Simple test: solve -∇²u = 1 with u=0 on boundary
domain = mesh.create_unit_square(comm, 4, 4)
V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary condition
def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    ])

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(1.0, v) * ufl.dx

# Method 1: LinearProblem (reference)
print("Method 1: LinearProblem")
problem = petsc.LinearProblem(a, L, bcs=[bc], 
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                              petsc_options_prefix="test_")
u1 = problem.solve()

# Method 2: Manual assembly
print("Method 2: Manual assembly")
a_form = fem.form(a)
L_form = fem.form(L)

A = petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()

b = petsc.create_vector(L_form.function_spaces)
with b.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b, L_form)
petsc.apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

u2 = fem.Function(V)
ksp.solve(b, u2.x.petsc_vec)
u2.x.scatter_forward()

# Compare
diff = u1.x.array - u2.x.array
print(f"Max difference: {np.max(np.abs(diff))}")

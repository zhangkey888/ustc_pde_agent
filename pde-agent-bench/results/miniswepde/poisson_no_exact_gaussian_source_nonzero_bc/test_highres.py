import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
N = 128
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
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
u_bc.interpolate(lambda x: np.full_like(x[0], 0.0))
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
f_expr = ufl.exp(-180 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_expr, v) * ufl.dx

a_form = fem.form(a)
L_form = fem.form(L)

A = petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = petsc.create_vector(L_form.function_spaces)

start = time.time()
with b.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b, L_form)
petsc.apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType('gmres')
ksp.getPC().setType('hypre')
ksp.setTolerances(rtol=1e-8)

u_sol = fem.Function(V)
ksp.solve(b, u_sol.x.petsc_vec)
u_sol.x.scatter_forward()
end = time.time()

print(f"Time for N={N}: {end-start:.3f}s")
print(f"Iterations: {ksp.getIterationNumber()}")

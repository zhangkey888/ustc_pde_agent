import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc

comm = MPI.COMM_WORLD

# Simple Poisson: -∇²u = 1, u=0 on boundary
domain = mesh.create_unit_square(comm, 10, 10)
V = fem.functionspace(domain, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, 1.0), v) * ufl.dx

# Dirichlet BC
def boundary(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ])

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(0.0, dofs, V)

# Solve with LinearProblem
problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="test_"
)
u_sol = problem.solve()

print(f"Solution min: {u_sol.x.array.min():.6f}, max: {u_sol.x.array.max():.6f}")
print("Should be positive with max at center")

# Now implement manually to understand
print("\nManual implementation:")
A_manual = petsc.assemble_matrix(fem.form(a), bcs=[bc])
A_manual.assemble()

b_manual = petsc.create_vector(fem.form(L).function_spaces)
with b_manual.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b_manual, fem.form(L))
petsc.apply_lifting(b_manual, [fem.form(a)], bcs=[[bc]])
b_manual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b_manual, [bc])

# Solve
from petsc4py import PETSc
solver = PETSc.KSP().create(comm)
solver.setOperators(A_manual)
solver.setType("preonly")
solver.getPC().setType("lu")

u_manual = fem.Function(V)
solver.solve(b_manual, u_manual.x.petsc_vec)
u_manual.x.scatter_forward()

print(f"Manual solution min: {u_manual.x.array.min():.6f}, max: {u_manual.x.array.max():.6f}")

# Compare
diff = u_sol.x.array - u_manual.x.array
print(f"Difference: max={np.max(np.abs(diff)):.6e}")

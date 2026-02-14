from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 8, 8)
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, PETSc.ScalarType(1.0)), v) * ufl.dx

problem = petsc.LinearProblem(a, L, petsc_options_prefix="test")
u_sol = problem.solve()
print("Type of problem:", type(problem))
print("Dir:", dir(problem))
print("\nHas solver?", hasattr(problem, 'solver'))
if hasattr(problem, 'solver'):
    print("Solver type:", type(problem.solver))
    print("Solver dir:", dir(problem.solver))

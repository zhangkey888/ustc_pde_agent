import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
f_expr = ufl.sin(12 * np.pi * x[0]) * ufl.sin(10 * np.pi * x[1])
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_expr, v) * ufl.dx

# Define BC
def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ])
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_bc, dofs)

problem = petsc.LinearProblem(
    a, L, bcs=[bc], 
    petsc_options={"ksp_type": "gmres", "pc_type": "hypre"},
    petsc_options_prefix="test_"
)
u_sol = problem.solve()

print(f"Type of problem.solver: {type(problem.solver)}")
print(f"Dir of problem.solver: {dir(problem.solver)}")

# Try PETSc KSP method
ksp = problem.solver
print(f"KSP type: {type(ksp)}")
print(f"KSP getIterationNumber: {ksp.getIterationNumber()}")

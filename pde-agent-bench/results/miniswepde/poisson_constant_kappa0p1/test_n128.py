import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
rank = comm.rank

exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
kappa = 0.1
source = lambda x: kappa * 2.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

N = 128
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: exact(x))
bc = fem.dirichletbc(u_bc, dofs)

f = fem.Function(V)
f.interpolate(lambda x: source(x))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

t0 = time.time()
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8}, petsc_options_prefix="poisson_")
u_h = problem.solve()
t1 = time.time()
iterations = problem.solver.getIterationNumber()

u_exact = fem.Function(V)
u_exact.interpolate(lambda x: exact(x))
error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
error_L2 = np.sqrt(fem.assemble_scalar(error_form))
if rank == 0:
    print(f"N={N}, L2 error: {error_L2}, time: {t1-t0:.3f}s, iterations: {iterations}")

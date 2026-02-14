import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

comm = MPI.COMM_WORLD
rank = comm.rank

domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
                               [8, 8], cell_type=mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1))

# Simple nonlinear test
u = fem.Function(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

# Nonlinear residual: u^3 - f = 0
f = fem.Constant(domain, ScalarType(1.0))
F = ufl.inner(u**3, v) * ufl.dx - ufl.inner(f, v) * ufl.dx

# Boundary condition
def boundary_marker(x):
    return np.isclose(x[0], 0.0)
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

problem = petsc.NonlinearProblem(F, u, bcs=[bc])
solver = nls.petsc.NewtonSolver(comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8

n_iter, converged = solver.solve(u)
if rank == 0:
    print(f"Newton solver test: iterations={n_iter}, converged={converged}")

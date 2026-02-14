import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

comm = MPI.COMM_WORLD
rank = comm.rank

# Simple test: solve -∇²u = f with u=0 on boundary, f = 2π²*sin(πx)*sin(πy)
# Exact solution: u = sin(πx)*sin(πy)

N = 16
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Define boundary condition
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
bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

# Source term: f = 2π²*sin(πx)*sin(πy)
f = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Solve using LinearProblem
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# Exact solution
u_exact = fem.Function(V)
def exact_sol(x):
    return np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
u_exact.interpolate(exact_sol)

# Compute error
error_expr = ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx
error_form = fem.form(error_expr)
error_sq = fem.assemble_scalar(error_form)
error = np.sqrt(error_sq)

if rank == 0:
    print(f"Error for Poisson problem: {error:.6e}")
    print(f"Expected to be small (should be < 1e-3 for N=16)")

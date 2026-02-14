import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Create mesh
N = 32
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Exact solution at t=0
def exact_solution(x):
    return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])

# Source term f = -∇²u = 5*pi²*sin(pi*x)*sin(2*pi*y) (since ∂u/∂t=0)
def source_term(x):
    return 5 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])

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
u_bc.interpolate(exact_solution)
bc = fem.dirichletbc(u_bc, dofs)

# Variational form: -∇²u = f  => (∇u, ∇v) = (f, v)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Create source function
f_func = fem.Function(V)
f_func.interpolate(source_term)
L = ufl.inner(f_func, v) * ufl.dx

# Solve
problem = petsc.LinearProblem(a, L, bcs=[bc], 
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                              petsc_options_prefix="test_")
u_sol = problem.solve()

# Compute error
u_exact = fem.Function(V)
u_exact.interpolate(exact_solution)

error_func = fem.Function(V)
error_func.x.array[:] = u_sol.x.array - u_exact.x.array
error_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(error_func, error_func) * ufl.dx)))

if rank == 0:
    print(f"Steady-state L2 error: {error_norm:.6e}")
    # Should be very small (machine precision) since we're solving with exact BCs

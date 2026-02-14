import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Create mesh
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test function
x = ufl.SpatialCoordinate(domain)
u_exact = ufl.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
f = -u_exact * (6400 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) - 160)  # -∇²u

# Variational problem: -∇²u = f
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary condition
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)))

def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    ])

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_bc, dofs)

# Solve
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# Compute error
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(lambda x: np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)))

error = u_sol.x.array - u_exact_fe.x.array
l2_error = np.sqrt(np.mean(error**2))

if rank == 0:
    print(f"Steady-state L2 error: {l2_error:.6e}")
    print(f"Should be very small (manufactured solution)")

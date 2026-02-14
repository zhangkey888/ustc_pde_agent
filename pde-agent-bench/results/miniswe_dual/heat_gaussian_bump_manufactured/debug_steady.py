import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

# Create mesh
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Exact solution at t=0
def u_exact(x):
    return np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x):
    """Source term for steady state: -∇·(κ∇u) = f"""
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-40 * r2)
    kappa = 1.0
    # ∇²u = u * (6400*r² - 160)
    # For steady: -κ∇²u = f, so f = -κ*u*(6400*r² - 160)
    return -kappa * u_val * (6400 * r2 - 160)

# Boundary condition
u_bc = fem.Function(V)
u_bc.interpolate(u_exact)

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
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
kappa = 1.0

a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Source term
f_fe = fem.Function(V)
f_fe.interpolate(f_source)
L = ufl.inner(f_fe, v) * ufl.dx

# Solve
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# Compute error
u_exact_fe = fem.Function(V)
u_exact_fe.interpolate(u_exact)

error = u_sol.x.array - u_exact_fe.x.array
l2_error = np.sqrt(np.mean(error**2))
print(f"Steady-state L2 error: {l2_error:.6e}")

# Check if solution is reasonable
print(f"Solution min: {u_sol.x.array.min():.6f}, max: {u_sol.x.array.max():.6f}")
print(f"Exact min: {u_exact_fe.x.array.min():.6f}, max: {u_exact_fe.x.array.max():.6f}")

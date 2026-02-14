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

# Solve using LinearProblem
problem = petsc.LinearProblem(
    a, L, 
    bcs=[bc], 
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="steady_"
)
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

# Also test the transient source term
def f_source_transient(x, t):
    """Source term for transient: f = ∂u/∂t - κ∇²u"""
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-t) * np.exp(-40 * r2)
    kappa = 1.0
    # ∂u/∂t = -u
    # ∇²u = u * (6400*r² - 160)
    # f = -u - κ*u*(6400*r² - 160) = -u*(1 + κ*(6400*r² - 160))
    return -u_val * (1 + kappa * (6400 * r2 - 160))

# Test at t=0.1
test_point = np.array([[0.5], [0.5], [0.0]])
f_val = f_source_transient(test_point, 0.1)
print(f"\nTest f at center, t=0.1: {f_val[0]:.6f}")

# What should it be? u(0.5,0.5,t) = exp(-t)
# At t=0.1, u = exp(-0.1) ≈ 0.904837
# ∇²u at center: r²=0, so ∇²u = u * (-160) = -160*u ≈ -144.774
# ∂u/∂t = -u ≈ -0.904837
# So f = -0.904837 - (-144.774) = 143.869
# But my formula gives: -u*(1 + (6400*0 - 160)) = -u*(1 - 160) = -u*(-159) = 159*u = 159*0.904837 ≈ 143.869
# Yes, matches.

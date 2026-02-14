import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType
comm = MPI.COMM_WORLD
rank = comm.rank

# Create mesh
N = 64
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

# Function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Define exact solution
x = ufl.SpatialCoordinate(domain)
u_exact_ufl = ufl.sin(np.pi * x[0] * x[1])

# κ = 1.0
kappa = fem.Constant(domain, ScalarType(1.0))

# Compute f = -∇·(κ ∇u_exact)
f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

# Convert to fem.Expression
f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
f_func = fem.Function(V)
f_func.interpolate(f_expr)

# Boundary condition
g_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
g_func = fem.Function(V)
g_func.interpolate(g_expr)

# Apply Dirichlet BC on entire boundary
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
bc = fem.dirichletbc(g_func, dofs)

# Variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_func, v) * ufl.dx

# Solve
problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000
    },
    petsc_options_prefix="test_"
)

u_sol = problem.solve()

# Compute L2 error
u_exact_func = fem.Function(V)
u_exact_func.interpolate(g_expr)

# Error function
error_func = fem.Function(V)
error_func.x.array[:] = u_sol.x.array - u_exact_func.x.array

# L2 norm of error
error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
error_norm = np.sqrt(fem.assemble_scalar(error_form))

if rank == 0:
    print(f"L2 error norm: {error_norm:.6e}")
    
    # Check against requirement
    if error_norm <= 1.08e-04:
        print(f"PASS: error {error_norm:.6e} <= 1.08e-04")
    else:
        print(f"FAIL: error {error_norm:.6e} > 1.08e-04")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType
comm = MPI.COMM_WORLD
rank = comm.rank

# Test with N=64 (what our solver chooses)
N = 64
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

# Function space with degree 1
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

# Compute error on 50x50 grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

points = np.zeros((3, nx * ny))
points[0, :] = X.flatten()
points[1, :] = Y.flatten()
points[2, :] = 0.0

u_grid_flat = np.full((nx * ny,), np.nan)

if rank == 0:
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    u_grid = u_grid_flat.reshape((nx, ny))
    u_exact_grid = np.sin(np.pi * X * Y)
    
    # Compute L2 error on grid (approximate integration)
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    error_sq = np.sum((u_grid - u_exact_grid)**2) * dx * dy
    error_l2_grid = np.sqrt(error_sq)
    
    # Compute max error
    max_error = np.max(np.abs(u_grid - u_exact_grid))
    
    print(f"N={N}, degree=1")
    print(f"L2 error on 50x50 grid: {error_l2_grid:.6e}")
    print(f"Max error on 50x50 grid: {max_error:.6e}")
    print(f"Accuracy requirement: ≤ 1.08e-04")
    
    if error_l2_grid <= 1.08e-04:
        print(f"PASS: Grid error {error_l2_grid:.6e} ≤ 1.08e-04")
    else:
        print(f"FAIL: Grid error {error_l2_grid:.6e} > 1.08e-04")

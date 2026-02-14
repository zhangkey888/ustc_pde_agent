import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

def exact_solution(x):
    return (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 
            0.2 * np.sin(5 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))

# Test with N=64 and degree 3
N = 64
degree = 3
start = time.time()
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Boundary conditions
tdim = domain.topology.dim
fdim = tdim - 1
def boundary_marker(x):
    return np.ones(x.shape[1], dtype=bool)
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: exact_solution(x))
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
pi = np.pi
f_expr = (2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 
          8.2 * pi**2 * ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1]))
f_func = fem.Function(V)
f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_func, v) * ufl.dx

problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="poisson_"
)
u_sol = problem.solve()

# Evaluate on 50x50 grid
nx = ny = 50
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
points = np.zeros((3, nx * ny))
points[0, :] = X.flatten()
points[1, :] = Y.flatten()
points[2, :] = 0.0

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

u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
if len(points_on_proc) > 0:
    vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
    u_values[eval_map] = vals.flatten()

u_grid = u_values.reshape((nx, ny))
exact_grid = exact_solution([X, Y, np.zeros_like(X)])

max_error = np.abs(u_grid - exact_grid).max()
l2_error = np.sqrt(np.mean((u_grid - exact_grid)**2))
end = time.time()
print(f"N={N}, degree={degree}, time={end-start:.3f}s")
print(f"Max error: {max_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
print(f"u min/max: {u_grid.min():.6f}, {u_grid.max():.6f}")

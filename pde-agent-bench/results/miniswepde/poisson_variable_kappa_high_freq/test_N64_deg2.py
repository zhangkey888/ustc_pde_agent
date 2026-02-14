import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

comm = MPI.COMM_WORLD
N = 64
degree = 2
domain = mesh.create_unit_square(comm, nx=N, ny=N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Boundary
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
u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
bc = fem.dirichletbc(u_bc, dofs)

x = ufl.SpatialCoordinate(domain)
u_exact = ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
kappa = 1.0 + 0.3 * ufl.sin(8 * np.pi * x[0]) * ufl.sin(8 * np.pi * x[1])
f_expr = -ufl.div(kappa * ufl.grad(u_exact))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_expr, v) * ufl.dx

problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
    petsc_options_prefix="pdebench_"
)
u_sol = problem.solve()

# Evaluate on 50x50 grid
nx = ny = 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).astype(ScalarType)

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

ref = np.load('oracle_output/reference.npz')
u_exact_grid = ref['u_star']
error = np.abs(u_grid - u_exact_grid)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print("Degree", degree, "N", N)
print("Max error:", max_error)
print("L2 error:", l2_error)
print("Pass?", max_error <= 1.28e-3)

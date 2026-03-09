import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

t_start = time.time()

E = 1.0
nu_val = 0.33
lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
mu = E / (2.0 * (1.0 + nu_val))

N = 48
degree = 3

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
gdim = domain.geometry.dim

V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

x = ufl.SpatialCoordinate(domain)
pi = ufl.pi

u_exact_ufl = ufl.as_vector([
    ufl.exp(2*x[0]) * ufl.sin(pi*x[1]),
    -ufl.exp(2*x[1]) * ufl.sin(pi*x[0])
])

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

f = -ufl.div(sigma(u_exact_ufl))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

u_bc = fem.Function(V)

def u_exact_func(x):
    vals = np.zeros((gdim, x.shape[1]))
    vals[0] = np.exp(2*x[0]) * np.sin(np.pi * x[1])
    vals[1] = -np.exp(2*x[1]) * np.sin(np.pi * x[0])
    return vals

u_bc.interpolate(u_exact_func)

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_bc, dofs)

problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": "1e-12",
        "ksp_atol": "1e-14",
        "ksp_max_it": "2000",
    },
    petsc_options_prefix="elasticity_"
)
u_sol = problem.solve()

elapsed_solve = time.time() - t_start

nx_eval, ny_eval = 50, 50
xs = np.linspace(0, 1, nx_eval)
ys = np.linspace(0, 1, ny_eval)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

points_2d = np.column_stack([XX.ravel(), YY.ravel()])
points_3d = np.zeros((points_2d.shape[0], 3))
points_3d[:, :2] = points_2d

bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

u_grid = np.full((nx_eval * ny_eval,), np.nan)
points_on_proc = []
cells_on_proc = []
eval_map = []

for i in range(len(points_3d)):
    links = colliding_cells.links(i)
    if len(links) > 0:
        points_on_proc.append(points_3d[i])
        cells_on_proc.append(links[0])
        eval_map.append(i)

if len(points_on_proc) > 0:
    pts_arr = np.array(points_on_proc)
    cells_arr = np.array(cells_on_proc, dtype=np.int32)
    vals = u_sol.eval(pts_arr, cells_arr)
    disp_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
    for idx, global_idx in enumerate(eval_map):
        u_grid[global_idx] = disp_mag[idx]

u_grid_2d = u_grid.reshape((nx_eval, ny_eval))

ux_exact = np.exp(2*XX) * np.sin(np.pi * YY)
uy_exact = -np.exp(2*YY) * np.sin(np.pi * XX)
mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

error = np.sqrt(np.nanmean((u_grid_2d - mag_exact)**2))
rel_error = error / np.sqrt(np.nanmean(mag_exact**2))
max_error = np.nanmax(np.abs(u_grid_2d - mag_exact))
elapsed = time.time() - t_start
print(f"N={N}, degree={degree}")
print(f"Wall time: {elapsed:.3f}s")
print(f"RMS error: {error:.2e}")
print(f"Relative RMS error: {rel_error:.2e}")
print(f"Max absolute error: {max_error:.2e}")

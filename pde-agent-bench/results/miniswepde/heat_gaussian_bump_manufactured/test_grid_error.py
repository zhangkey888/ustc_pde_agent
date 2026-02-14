import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    u = u_exact(x, t)
    du_dt = -u
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    laplacian_u = u * (-80 + 1600 * r2)
    return du_dt - laplacian_u

N = 64
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 2))  # degree 2

# Boundary condition dofs
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

t_end = 0.1
u_exact_func = fem.Function(V)
u_exact_func.interpolate(lambda x: u_exact(x, t_end))

f_func = fem.Function(V)
f_func.interpolate(lambda x: f_source(x, t_end))

# Solve -Δu = f + u_exact
g_func = fem.Function(V)
g_func.x.array[:] = f_func.x.array + u_exact_func.x.array

# Solve -Δu = g
u = fem.Function(V)
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
L = ufl.inner(g_func, v) * ufl.dx

# Dirichlet BC
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: u_exact(x, t_end))
bc = fem.dirichletbc(u_bc, dofs)

problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="test_")
u_sol = problem.solve()

# Sample on 50x50 grid
nx = ny = 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T

# Evaluate solution at points
u_grid = np.zeros((nx, ny))
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

points_on_proc = []
cells_on_proc = []
eval_map = []

for i in range(points.shape[0]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        points_on_proc.append(points[i])
        cells_on_proc.append(links[0])
        eval_map.append(i)

if len(points_on_proc) > 0:
    vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
    flat_vals = vals.flatten()
    linear_indices = np.unravel_index(eval_map, (nx, ny))
    for idx, val in zip(zip(*linear_indices), flat_vals):
        u_grid[idx] = val

# Gather on rank 0
if comm.size > 1:
    u_grid_full = np.zeros_like(u_grid)
    comm.Reduce(u_grid, u_grid_full, op=MPI.SUM, root=0)
    if rank == 0:
        u_grid = u_grid_full
    else:
        u_grid = np.zeros((nx, ny))

if rank == 0:
    # Compute exact on grid
    exact_grid = u_exact(np.array([X, Y]), t_end)
    error_grid = np.abs(u_grid - exact_grid)
    max_error = np.max(error_grid)
    rms_error = np.sqrt(np.mean(error_grid**2))
    print(f"Max error on 50x50 grid: {max_error:.6e}")
    print(f"RMS error on grid: {rms_error:.6e}")
    print(f"Required accuracy: 2.49e-03")
    # Check center
    center_val = u_grid[nx//2, ny//2]
    exact_center = u_exact(np.array([0.5, 0.5]), t_end)
    print(f"Center: num={center_val:.6f}, exact={exact_center:.6f}, diff={abs(center_val-exact_center):.6e}")

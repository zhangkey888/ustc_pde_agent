import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

# Test with N=128 directly
N = 128
element_degree = 1

start_time = time.time()

# Create mesh
domain = mesh.create_unit_square(comm, nx=N, ny=N, cell_type=mesh.CellType.triangle)

# Function space
V = fem.functionspace(domain, ("Lagrange", element_degree))

# Define exact solution and source term
x = ufl.SpatialCoordinate(domain)
u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
f = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

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
u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, ScalarType(1.0)) * f, v) * ufl.dx

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
    petsc_options_prefix="poisson_"
)

u_sol = problem.solve()

# Evaluate on 50x50 grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
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

u_grid = u_values.reshape(nx, ny)

end_time = time.time()

print(f"Time taken: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution: {N}")

# Compute error
u_exact_grid = np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(u_grid - u_exact_grid)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy requirement: ≤ 5.81e-04")
print(f"Time requirement: ≤ 2.131s")
print(f"Pass accuracy: {max_error <= 5.81e-04}")
print(f"Pass time: {(end_time - start_time) <= 2.131}")

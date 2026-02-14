import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 8, 8, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# Parameters
kappa = 10.0
dt = 0.005
t = 0.0

# Exact solution at t=0 and t=dt
u_exact_0 = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_exact_dt = ufl.exp(-dt) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])

# BC at t=dt
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
u_bc.interpolate(fem.Expression(u_exact_dt, V.element.interpolation_points))
bc = fem.dirichletbc(u_bc, dofs)

# Initial condition u_n
u_n = fem.Function(V)
u_n.interpolate(fem.Expression(u_exact_0, V.element.interpolation_points))

# Source term at t=0? Actually f depends on time. Use f at t=0 for backward Euler from t=0 to dt?
# f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi**2)
f_expr = ufl.exp(-0.0) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
f_func = fem.Function(V)
f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

# Variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx

# Solve
problem = petsc.LinearProblem(a, L, bcs=[bc],
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                              petsc_options_prefix="verify_")
u_sol = problem.solve()

# Compute error
u_exact = fem.Function(V)
u_exact.interpolate(fem.Expression(u_exact_dt, V.element.interpolation_points))
error = fem.Function(V)
error.x.array[:] = u_sol.x.array - u_exact.x.array
error_form = fem.form(ufl.inner(error, error) * ufl.dx)
error_norm = np.sqrt(fem.assemble_scalar(error_form))
print(f"Error after one time step (coarse mesh): {error_norm:.2e}")

# Print values at center
import dolfinx.geometry
points = np.array([[0.5, 0.5, 0.0]])
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
cells = []
for i in range(points.shape[0]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        cells.append(links[0])
val_sol = u_sol.eval(points, np.array(cells, dtype=np.int32))
val_exact = u_exact.eval(points, np.array(cells, dtype=np.int32))
print(f"Numerical u at center: {val_sol[0]:.6f}, exact: {val_exact[0]:.6f}")
print(f"Ratio: {val_sol[0]/val_exact[0]:.6f}")

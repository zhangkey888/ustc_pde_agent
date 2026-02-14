import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 8, 8, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# Test exact solution at t=0
u_exact_expr = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_exact = fem.Function(V)
u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

# Evaluate at center
points = np.array([[0.5, 0.5, 0.0]])
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
cells = []
for i in range(points.shape[0]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        cells.append(links[0])
val = u_exact.eval(points, np.array(cells, dtype=np.int32))
print(f"Exact u at center (t=0): {val[0]:.6f}, expected: {np.sin(pi*0.5)*np.sin(pi*0.5):.6f}")

# Test source term at t=0, kappa=10
kappa = 10.0
f_expr = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
f_func = fem.Function(V)
f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
val_f = f_func.eval(points, np.array(cells, dtype=np.int32))
print(f"Source f at center (t=0): {val_f[0]:.6f}, expected: {np.sin(pi*0.5)*np.sin(pi*0.5)*(-1 + 2*kappa*pi**2):.6f}")
print(f"(-1 + 2*kappa*pi**2) = {-1 + 2*kappa*pi**2}")

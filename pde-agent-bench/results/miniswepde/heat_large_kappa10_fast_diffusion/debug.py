import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# Exact solution at t=0.05
t_end = 0.05
u_exact_expr = ufl.exp(-t_end) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_exact = fem.Function(V)
u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

# Evaluate at center using eval
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
val = u_exact.eval(points, np.array(cells, dtype=np.int32))
print(f"Exact u at center (t=0.05): {val[0]:.6f}, expected: {np.exp(-0.05)*np.sin(pi*0.5)*np.sin(pi*0.5):.6f}")

# Also compute max over mesh vertices
print(f"Exact u max over vertices: {np.max(u_exact.x.array):.6f}")

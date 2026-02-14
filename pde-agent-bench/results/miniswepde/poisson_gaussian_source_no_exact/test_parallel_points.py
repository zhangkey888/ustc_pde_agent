import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc

comm = MPI.COMM_WORLD
rank = comm.rank

domain = mesh.create_unit_square(comm, 16, 16)
V = fem.functionspace(domain, ("Lagrange", 1))
u = fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])

nx = ny = 5
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T

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

u_values = np.full((points.shape[0],), np.nan)
if len(points_on_proc) > 0:
    vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
    u_values[eval_map] = vals.flatten()

# Gather to root
if rank == 0:
    gathered = np.empty((comm.size, points.shape[0]), dtype=np.float64)
else:
    gathered = None
comm.Gather(u_values, gathered, root=0)

if rank == 0:
    # Combine: take first non-nan value for each point
    combined = np.nanmax(gathered, axis=0)
    print("Combined shape:", combined.shape)
    print("Number of nan:", np.sum(np.isnan(combined)))
    print("Sample values:", combined[:5])
else:
    combined = None

# Broadcast combined to all processes
combined = comm.bcast(combined, root=0)
print(f"Rank {rank}: combined min/max", np.nanmin(combined), np.nanmax(combined))

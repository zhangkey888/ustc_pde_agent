import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test exact solution interpolation
def exact_solution(x):
    # x has shape (3, N)
    return np.exp(-0.08) * np.exp(5*x[1]) * np.sin(np.pi*x[0])

u_exact_func = fem.Function(V)
u_exact_func.interpolate(exact_solution)

# Test source term
def source_term(x):
    pi = np.pi
    return np.exp(-0.08) * np.exp(5*x[1]) * np.sin(pi*x[0]) * (pi**2 - 26)

f_func = fem.Function(V)
f_func.interpolate(source_term)

print("Interpolation test completed")
print(f"u_exact min/max: {u_exact_func.x.array.min():.3f}, {u_exact_func.x.array.max():.3f}")
print(f"f min/max: {f_func.x.array.min():.3f}, {f_func.x.array.max():.3f}")

# Check at a specific point
import dolfinx.geometry
points = np.array([[0.5], [0.5], [0.0]])  # shape (3, 1)
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points.T)

for i in range(points.shape[1]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        val = u_exact_func.eval(points.T[i], np.array([links[0]], dtype=np.int32))
        print(f"u_exact at (0.5, 0.5): {val[0]:.3f}")
        val_f = f_func.eval(points.T[i], np.array([links[0]], dtype=np.int32))
        print(f"f at (0.5, 0.5): {val_f[0]:.3f}")
        # Compute exact value manually
        manual = np.exp(-0.08) * np.exp(5*0.5) * np.sin(np.pi*0.5)
        print(f"Manual u_exact: {manual:.3f}")
        manual_f = manual * (np.pi**2 - 26)
        print(f"Manual f: {manual_f:.3f}")

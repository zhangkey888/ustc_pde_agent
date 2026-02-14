import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 10, 10)
V = fem.functionspace(domain, ("Lagrange", 1))

def boundary_marker(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ])

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

print(f"Number of boundary facets: {len(boundary_facets)}")
print(f"Number of boundary DOFs: {len(dofs)}")

# Create a function with BC
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.ones_like(x[0]))  # Constant 1 on boundary

bc = fem.dirichletbc(u_bc, dofs)

# Test by solving Laplace equation with this BC
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, 0.0), v) * ufl.dx

from dolfinx.fem import petsc
problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# Check boundary values
print("Checking boundary values...")
# Evaluate at boundary points
points = np.array([[0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

from dolfinx import geometry
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

for i in range(points.shape[1]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        val = u_sol.eval(points.T[i:i+1], np.array([links[0]], dtype=np.int32))
        print(f"Point ({points[0,i]:.1f}, {points[1,i]:.1f}): value = {val[0]:.6f}")

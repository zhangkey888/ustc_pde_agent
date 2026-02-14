import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve_poisson(N=128, degree=1):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Kappa
    kappa = fem.Function(V)
    kappa.interpolate(lambda x: 1.0 + 0.4 * np.cos(4 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    
    # Exact solution and source term
    u_exact_expr = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    kappa_expr = 1.0 + 0.4 * ufl.cos(4 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_expr)
    kappa_grad_u = kappa_expr * grad_u_exact
    f_expr = -ufl.div(kappa_grad_u)
    
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Variational form
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
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
    
    g = fem.Function(V)
    g.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(g, dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="pde_"
    )
    u_h = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)])
    
    # Evaluate function
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid

# Test with N=128, degree=1
print("Testing N=128, degree=1:")
u_grid = solve_poisson(N=128, degree=1)

# Compute error
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u_grid - u_exact)
max_error = np.max(error)
print(f"Max error: {max_error:.6e}")
print(f"Accuracy requirement: ≤ 2.76e-04")
print(f"Pass: {max_error <= 2.76e-04}")

# Test with N=64, degree=2
print("\nTesting N=64, degree=2:")
u_grid2 = solve_poisson(N=64, degree=2)
error2 = np.abs(u_grid2 - u_exact)
max_error2 = np.max(error2)
print(f"Max error: {max_error2:.6e}")
print(f"Accuracy requirement: ≤ 2.76e-04")
print(f"Pass: {max_error2 <= 2.76e-04}")

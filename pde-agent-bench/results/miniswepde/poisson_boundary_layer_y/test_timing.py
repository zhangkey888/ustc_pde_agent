import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def exact_solution(x):
    return np.exp(6 * x[1]) * np.sin(np.pi * x[0])

def evaluate_function_at_points(u_func, points):
    domain = u_func.function_space.mesh
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
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    if np.any(np.isnan(u_values)):
        comm = MPI.COMM_WORLD
        all_values = comm.allgather(u_values)
        u_values = np.nanmean(np.array(all_values), axis=0)
    return u_values

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType
pi = np.pi

configs = [(64, 2), (96, 2), (128, 2), (64, 3)]
for N, degree in configs:
    if comm.rank == 0:
        print(f"\nTesting N={N}, degree={degree}")
    start = time.time()
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
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
    u_bc.interpolate(exact_solution)
    bc = fem.dirichletbc(u_bc, dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    κ = fem.Constant(domain, ScalarType(1.0))
    x = ufl.SpatialCoordinate(domain)
    f_expr = (pi**2 - 36) * ufl.exp(6 * x[1]) * ufl.sin(pi * x[0])
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_function = fem.Function(V)
    f_function.interpolate(f)
    
    a = ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_function, v) * ufl.dx
    
    u_sol = fem.Function(V)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="test_"
    )
    u_sol = problem.solve()
    
    # Compute error
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    u_vals = evaluate_function_at_points(u_sol, points)
    exact_vals = exact_solution(points)
    max_error = np.max(np.abs(u_vals - exact_vals))
    
    end = time.time()
    if comm.rank == 0:
        print(f"  Max error: {max_error:.2e}, Time: {end-start:.2f}s")

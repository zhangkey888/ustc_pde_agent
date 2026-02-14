import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve_degree1():
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    N = 64  # Use the resolution that converged
    element_degree = 1  # Linear elements
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define boundary condition
    def u_exact_func(x):
        return np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
    
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
    u_bc.interpolate(u_exact_func)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 0.5 * ufl.sin(6 * np.pi * x[0])
    
    u_exact = ufl.sin(2*np.pi*x[0]) * ufl.sin(np.pi*x[1])
    grad_u = ufl.grad(u_exact)
    f_expr = -ufl.div(kappa_expr * grad_u)
    
    f = fem.Function(V)
    f_expr_compiled = fem.Expression(f_expr, V.element.interpolation_points)
    f.interpolate(f_expr_compiled)
    
    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    u_sol = fem.Function(V)
    
    # Use direct solver for reliability
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Simple evaluation (assuming 1 process)
    domain = u_sol.function_space.mesh
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Compute error
    u_exact_grid = np.sin(2*np.pi*X) * np.sin(np.pi*Y)
    error_grid = u_grid - u_exact_grid
    l2_error = np.sqrt(np.mean(error_grid**2))
    
    return l2_error, u_grid.shape

if __name__ == "__main__":
    start = time.time()
    l2_error, shape = solve_degree1()
    end = time.time()
    
    print(f"Degree 1 solve time: {end - start:.3f} seconds")
    print(f"L2 error with degree 1: {l2_error:.6e}")
    print(f"Required accuracy: ≤ 7.47e-04")
    print(f"Accuracy requirement met: {l2_error <= 7.47e-04}")

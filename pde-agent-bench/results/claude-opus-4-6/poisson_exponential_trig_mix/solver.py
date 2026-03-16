import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(2 * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # kappa = 1.0, so f = -div(kappa * grad(u_exact)) = -laplacian(u_exact)
    # u = exp(2x)*cos(pi*y)
    # u_xx = 4*exp(2x)*cos(pi*y)
    # u_yy = -pi^2*exp(2x)*cos(pi*y)
    # laplacian = (4 - pi^2)*exp(2x)*cos(pi*y)
    # f = -laplacian = (pi^2 - 4)*exp(2x)*cos(pi*y)
    f_expr = (ufl.pi**2 - 4.0) * ufl.exp(2 * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Boundary conditions
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.exp(2 * x[0]) * np.cos(np.pi * x[1]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_monitor": "",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = X.ravel()
    points[1] = Y.ravel()
    
    # Point evaluation
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
    
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(iterations),
        }
    }
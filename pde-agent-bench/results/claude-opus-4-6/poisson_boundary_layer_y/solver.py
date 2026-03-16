import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # The manufactured solution is u = exp(6*y)*sin(pi*x)
    # With kappa=1, f = -laplacian(u) = -(d^2u/dx^2 + d^2u/dy^2)
    # d^2u/dx^2 = -pi^2 * exp(6y) * sin(pi*x)
    # d^2u/dy^2 = 36 * exp(6y) * sin(pi*x)
    # So -laplacian(u) = (pi^2 - 36) * exp(6y) * sin(pi*x)
    # f = (pi^2 - 36) * exp(6y) * sin(pi*x)
    
    # Boundary layer in y direction due to exp(6*y) — need good resolution
    # exp(6) ≈ 403, so large gradients near y=1
    
    # Use degree 2 elements for better accuracy
    degree = 2
    nx = ny = 64
    
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Source term
    f_expr = (ufl.pi**2 - 36.0) * ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions - apply exact solution on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.ravel()
    points[1] = Y.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }
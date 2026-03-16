import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    epsilon = pde_config.get("epsilon", 0.03)
    beta = pde_config.get("beta", [5.0, 2.0])
    
    # High Peclet number (~179.5), need SUPG stabilization
    # Use P2 elements for better accuracy
    element_degree = 2
    nx = ny = 80  # mesh resolution
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(pi*x)*sin(2*pi*y) = -pi^2*sin(pi*x)*sin(2*pi*y) - 4*pi^2*sin(pi*x)*sin(2*pi*y)
    #                                     = -5*pi^2*sin(pi*x)*sin(2*pi*y)
    # So -eps * laplacian = eps * 5 * pi^2 * sin(pi*x)*sin(2*pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # beta . grad(u) = 5*pi*cos(pi*x)*sin(2*pi*y) + 2*2*pi*sin(pi*x)*cos(2*pi*y)
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f_expr = (epsilon * 5.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
              + beta[1] * 2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1]))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_gal = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Simpler: tau = h^2 / (4*epsilon + 2*|beta|*h)  -- doubly asymptotic
    tau = h**2 / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG: residual applied to test function beta . grad(v)
    # For linear elements, laplacian of u_h = 0, but for P2 it's nonzero within elements
    # Residual: -eps*laplacian(u) + beta.grad(u) - f
    # For trial function approach (consistent linearization):
    # SUPG test function modification: v -> v + tau * beta.grad(v)
    
    r_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    # SUPG additional terms: inner(residual, tau * beta.grad(v))
    # Since we have a linear problem, the residual of trial function is:
    # -eps * div(grad(u)) + beta.grad(u) - f
    # For P2 elements, -eps*div(grad(u)) is piecewise constant (nonzero)
    a_supg = (ufl.inner(-epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)), r_test)) * ufl.dx
    L_supg = ufl.inner(f_expr, r_test) * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # Boundary conditions: u = 0 on all boundaries (since sin(pi*x)*sin(2*pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_eval * n_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
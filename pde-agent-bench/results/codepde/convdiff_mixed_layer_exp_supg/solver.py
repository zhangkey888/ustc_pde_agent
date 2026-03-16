import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    params = pde_config.get("params", {})
    epsilon = params.get("epsilon", 0.01)
    beta = params.get("beta", [12.0, 0.0])
    
    # 2. Create mesh - use higher resolution for high Peclet number
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(3*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -epsilon * laplacian(u) + beta . grad(u)
    # u = exp(3x)*sin(pi*y)
    # du/dx = 3*exp(3x)*sin(pi*y)
    # du/dy = exp(3x)*pi*cos(pi*y)
    # d2u/dx2 = 9*exp(3x)*sin(pi*y)
    # d2u/dy2 = -pi^2*exp(3x)*sin(pi*y)
    # laplacian = (9 - pi^2)*exp(3x)*sin(pi*y)
    # f = -epsilon*(9 - pi^2)*exp(3x)*sin(pi*y) + beta[0]*3*exp(3x)*sin(pi*y) + beta[1]*exp(3x)*pi*cos(pi*y)
    
    f_expr = (-epsilon * (9.0 - ufl.pi**2) * ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * 3.0 * ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.exp(3.0 * x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1]))
    
    # Velocity vector
    beta_vec = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # 4. SUPG stabilized variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_gal = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # Use coth(Pe) - 1/Pe approximation, but simpler: just use h/(2*|beta|) scaled
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG test function modification: v_supg = beta . grad(v) * tau
    # Residual of the PDE applied to trial function: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements, but for P2 it's not zero
    # Full residual: -eps * div(grad(u)) + beta . grad(u) - f
    
    # SUPG additional terms
    r_u = -eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    supg_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(r_u, supg_test) * ufl.dx
    L_supg = f_expr * supg_test * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create exact solution function for BC
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract on 50x50 grid
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_eval * n_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
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
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }
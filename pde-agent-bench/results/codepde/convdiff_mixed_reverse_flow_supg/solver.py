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
    epsilon = params.get("epsilon", 0.005)
    beta = params.get("beta", [-20.0, 5.0])
    
    # High Peclet number => need SUPG stabilization and fine mesh
    N = 128
    degree = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinate and exact solution
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = exp(x)*sin(pi*y)
    u_exact = ufl.exp(x[0]) * ufl.sin(pi * x[1])
    
    # Compute source term: f = -eps * laplacian(u) + beta . grad(u)
    # grad(u_exact) = (exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    # laplacian(u_exact) = exp(x)*sin(pi*y) + exp(x)*(-pi^2)*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi^2)
    # So -eps * laplacian = -eps * exp(x)*sin(pi*y)*(1 - pi^2) = eps*(pi^2 - 1)*exp(x)*sin(pi*y)
    # beta . grad(u) = beta[0]*exp(x)*sin(pi*y) + beta[1]*exp(x)*pi*cos(pi*y)
    
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    
    beta_vec = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    f_expr = -eps_const * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_gal = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
             ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_gal = ufl.inner(f_expr, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_local = beta_norm * h / (2.0 * eps_const)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ~ h/(2|beta|) for large Pe
    # Simplified: for high Peclet
    tau = h / (2.0 * beta_norm + 1e-10)
    
    # SUPG: add tau * (beta . grad(v)) * residual
    # Residual of strong form applied to trial: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 on each element
    # So residual ~ beta.grad(u) - f
    r_u = ufl.dot(beta_vec, ufl.grad(u)) - f_expr  # strong residual (linear part)
    
    a_supg = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(v)), ufl.dot(beta_vec, ufl.grad(u))) * ufl.dx
    L_supg = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(v)), f_expr) * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # 7. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    solver = problem.solver
    iterations = solver.getIterationNumber()
    
    # 8. Extract on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "stabilization": "SUPG",
        }
    }
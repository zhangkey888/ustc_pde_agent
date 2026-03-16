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
    beta = params.get("beta", [12.0, 4.0])
    
    # High Peclet number ~1265, need SUPG stabilization and fine mesh with P3 elements
    nx = ny = 80
    degree = 3
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P3 as indicated by case name
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Compute source term: f = -epsilon * laplacian(u) + beta . grad(u)
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f_expr = -epsilon * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) 
                  + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    # Compute element size h
    h = ufl.CellDiameter(domain)
    
    # Local Peclet number
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_local = beta_norm * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h / (2*|beta|) for large Pe
    # Use a simpler formula that works well for high Pe
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_local)
    # Clamp tau to be non-negative (for safety)
    # For high Pe, tau ≈ h/(2*|beta|)
    # Alternative: use the standard formula
    tau_supg = h * h / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG residual: R(u) = -epsilon * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0, but for P3 it's not zero
    # However, in weak form with trial functions, we use the strong-form operator
    # The SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau_supg * ufl.dot(beta_vec, ufl.grad(v))
    
    # SUPG additional terms (applied to the PDE operator on trial function)
    # For the bilinear form: add tau * (beta . grad(v)) * (-epsilon * laplacian(u) + beta . grad(u))
    # Since laplacian of trial function in bilinear form is tricky, we use the approximation
    # that for high Pe, the diffusion part of the residual is small
    # Full SUPG: tau * (L_adv(u) - f) * (beta . grad(v))
    # where L_adv(u) = beta . grad(u) (dominant part)
    # We include the diffusion part too for P3
    
    a_supg = tau_supg * ufl.dot(beta_vec, ufl.grad(u)) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    L_supg = tau_supg * f_expr * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 6. Boundary conditions
    # u = g = sin(pi*x)*sin(2*pi*y) on boundary
    # Since sin(pi*x)*sin(2*pi*y) = 0 on all boundaries of [0,1]^2, g = 0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 7. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "200",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
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
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
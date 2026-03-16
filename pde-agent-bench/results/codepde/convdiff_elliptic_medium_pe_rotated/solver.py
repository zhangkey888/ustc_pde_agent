import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    epsilon = pde_config.get("epsilon", 0.05)
    beta = pde_config.get("beta", [3.0, 1.0])
    
    # 2. Create mesh - use higher resolution for accuracy with high Peclet number
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use degree 2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    pi = ufl.pi
    u_exact = ufl.sin(2 * pi * (x[0] + x[1])) * ufl.sin(pi * (x[0] - x[1]))
    
    # Compute source term from manufactured solution
    # f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f = -epsilon * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 4. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    h = ufl.CellDiameter(domain)
    
    # Standard Galerkin terms
    a_standard = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L_standard = f * v * ufl.dx
    
    # SUPG stabilization
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_local = beta_norm * h / (2.0 * epsilon)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ~ h/(2|beta|) for large Pe
    # Simplified: tau = h / (2 * |beta|) * min(1, Pe/3)
    # Or use the standard formula:
    tau_supg = h / (2.0 * beta_norm + 1e-10) * ufl.min_value(1.0, Pe_local / 3.0)
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    r_test = tau_supg * ufl.dot(beta_vec, ufl.grad(v))
    
    # Residual applied to trial function: -eps*laplacian(u) + beta.grad(u)
    # For linear elements, laplacian of u is zero within elements, but for degree 2 it's not
    # We use the strong form residual: -eps * div(grad(u)) + beta . grad(u) - f
    # SUPG adds: integral of (residual) * tau * beta.grad(v)
    a_supg = (-epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))) * r_test * ufl.dx
    L_supg = f * r_test * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
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
    iterations = problem.solver.getIterationNumber()
    
    # 7. Extract on 50x50 grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
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
            "iterations": iterations,
        }
    }
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.005)
    beta_val = pde_config.get("beta", [18.0, 6.0])
    
    # High Peclet number problem - need SUPG stabilization and fine mesh
    # Pe ~ 3794.7, so we need good stabilization
    
    nx = ny = 128
    degree = 1
    
    # 2. Create mesh - quadrilateral as specified in case name
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Diffusion coefficient and velocity
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = ufl.as_vector([PETSc.ScalarType(beta_val[0]), PETSc.ScalarType(beta_val[1])])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u_exact) + beta . grad(u_exact) = f
    # u_exact = sin(pi*x)*sin(pi*y)
    # laplacian = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # -eps * (-2*pi^2 * sin(pi*x)*sin(pi*y)) = 2*eps*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u_exact) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    f_expr = (eps_val * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_val[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_val[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # 4. Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin forms
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # Peclet number based stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Optimal SUPG parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG residual: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements on quads, laplacian of u_h is zero within elements
    # So residual ≈ beta.grad(u) - f
    R_u = ufl.dot(beta, ufl.grad(u)) - f_expr
    
    # SUPG test function modification
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(R_u, v_supg) * ufl.dx
    # Split into bilinear and linear parts
    # R_u = beta.grad(u) - f
    # a_supg contribution: tau * (beta.grad(u)) * (beta.grad(v)) dx
    # L_supg contribution: tau * f * (beta.grad(v)) dx
    
    a_stab = tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_stab = tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    
    # Also add diffusion part of residual for higher accuracy (crosswind)
    # For P1 elements, -eps*laplacian(u_h) = 0 inside elements, so skip
    
    a_total = a_gal + a_stab
    L_total = L_gal + L_stab
    
    # 5. Boundary conditions - u = sin(pi*x)*sin(pi*y) = 0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
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
    
    # 7. Extract solution on 50x50 grid
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
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    epsilon = pde_config.get("epsilon", 0.01)
    beta = pde_config.get("beta", [12.0, 6.0])
    
    # 2. Create mesh - use fine mesh for high Peclet number
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1 with SUPG stabilization
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    pi = ufl.pi
    f_expr = (ufl.sin(8 * pi * x[0]) * ufl.sin(6 * pi * x[1]) 
              + 0.3 * ufl.sin(12 * pi * x[0]) * ufl.sin(10 * pi * x[1]))
    
    # Convection velocity
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Diffusion coefficient
    eps_c = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Standard Galerkin terms
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2*|beta|) for large Pe
    # Use a simpler robust formula:
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp tau to be non-negative by using a conditional-like approach
    # For high Peclet, Pe_cell >> 1 so tau ≈ h/(2*|beta|)
    # Alternative robust tau:
    tau = h * h / (4.0 * eps_c + 2.0 * h * beta_norm)
    
    # SUPG test function modification: v_supg = beta . grad(v) * tau
    # Residual of strong form applied to trial: -eps*laplacian(u) + beta.grad(u)
    # For linear elements, laplacian(u) = 0, so strong residual of LHS = beta.grad(u)
    # Strong residual of RHS = f
    
    r_lhs = ufl.dot(beta_vec, ufl.grad(u))  # -eps*laplacian(u) is zero for P1
    r_rhs = f_expr
    
    supg_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(r_lhs, supg_test) * ufl.dx
    L_supg = ufl.inner(r_rhs, supg_test) * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions - homogeneous Dirichlet (g=0 by default)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
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
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
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
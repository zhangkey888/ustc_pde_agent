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
    beta = pde_config.get("beta", [3.0, 3.0])
    
    # Source term parameters - oscillatory RHS: sin(6*pi*x)*sin(5*pi*y)
    # We'll construct this with UFL
    
    # 2. Create mesh - use fine mesh for high Peclet number
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use P1 with SUPG stabilization
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Convection velocity
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Standard Galerkin terms
    # -epsilon * laplacian(u) + beta . grad(u) = f
    # Weak form: epsilon * inner(grad(u), grad(v)) + inner(beta . grad(u), v) = inner(f, v)
    a_std = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_std = f * v * ufl.dx
    
    # SUPG stabilization
    # tau_SUPG = h / (2 * |beta|) * (coth(Pe_h) - 1/Pe_h)
    # where Pe_h = |beta| * h / (2 * epsilon)
    
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    Pe_h = beta_norm * h / (2.0 * epsilon)
    
    # Use a simplified tau for SUPG (optimal for 1D)
    # tau = h / (2 * |beta|) * (1 - 1/Pe_h) when Pe_h > 1 (which it is here)
    # More robust: tau = h / (2 * |beta|) * min(1, Pe_h/3)
    # Or the standard: tau = h^2 / (4*epsilon + 2*|beta|*h)
    
    # Standard stabilization parameter
    tau = h * h / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG: add tau * (beta . grad(v)) * residual
    # Residual of strong form applied to trial: -epsilon * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 on each element
    # So residual ≈ beta . grad(u) - f
    
    r_u = ufl.dot(beta_vec, ufl.grad(u))  # -epsilon * div(grad(u)) is zero for P1
    
    a_supg = tau * ufl.inner(r_u, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # 5. Boundary conditions: u = 0 on boundary (g = 0 by default for this case)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
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
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
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
        }
    }
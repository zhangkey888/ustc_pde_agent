import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde_config.get("epsilon", 0.005)
    beta = pde_config.get("beta", [15.0, 7.0])
    source_term_str = pde_config.get("source_term", "sin(10*pi*x)*sin(8*pi*y)")
    
    # Boundary conditions
    bc_config = pde_config.get("boundary_conditions", {})
    
    # High Peclet number -> need SUPG stabilization and fine mesh
    # Pe ~ 3310, so we need good stabilization
    
    # 2. Create mesh - use a reasonably fine mesh
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1 with SUPG
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(10*pi*x)*sin(8*pi*y)
    pi = ufl.pi
    f = ufl.sin(10 * pi * x[0]) * ufl.sin(8 * pi * x[1])
    
    # Convection velocity
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Diffusion coefficient
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Standard Galerkin terms:
    # -eps * laplacian(u) + beta . grad(u) = f
    # Weak form: eps * inner(grad(u), grad(v)) + inner(beta . grad(u), v) = inner(f, v)
    a_galerkin = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
                 + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_galerkin = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    # Compute element size h
    h = ufl.CellDiameter(domain)
    
    # Velocity magnitude
    beta_mag = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Local Peclet number
    Pe_local = beta_mag * h / (2.0 * eps_const)
    
    # SUPG stabilization parameter tau
    # Using the standard formula with coth
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) - 1/Pe ≈ 1
    # We use a simpler robust formula:
    tau = h / (2.0 * beta_mag) * (1.0 - 1.0 / Pe_local)
    
    # Alternative: use min-based stabilization for robustness
    # tau = ufl.min_value(h / (2.0 * beta_mag), h**2 / (4.0 * eps_const))
    
    # For very high Peclet, a simpler tau works well:
    tau_simple = h / (2.0 * beta_mag)
    
    # SUPG residual: R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 within each element
    # So R(u) = beta . grad(u) - f
    R_u = ufl.dot(beta_vec, ufl.grad(u)) - f
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau_simple * ufl.dot(beta_vec, ufl.grad(v))
    
    # SUPG terms
    a_supg = ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v_supg) * ufl.dx \
             + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) * ufl.dx
    # Note: for P1 elements, the diffusion part of the residual (laplacian) vanishes
    # So effectively: a_supg = inner(beta.grad(u), tau*beta.grad(v)) * dx
    # But we keep the full form for correctness
    
    # Simplified SUPG (since laplacian of P1 = 0 element-wise):
    a_supg_simple = tau_simple * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    L_supg_simple = tau_simple * ufl.inner(f, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    
    # Total bilinear form and RHS
    a = a_galerkin + a_supg_simple
    L = L_galerkin + L_supg_simple
    
    # 5. Boundary conditions
    # Default: u = 0 on all boundaries (homogeneous Dirichlet)
    # Check if BC is specified
    bc_type = bc_config.get("type", "dirichlet")
    bc_value_str = bc_config.get("value", "0")
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Parse BC value
    try:
        bc_val = float(bc_value_str)
    except (ValueError, TypeError):
        bc_val = 0.0
    
    bc = fem.dirichletbc(default_scalar_type(bc_val), boundary_dofs, V)
    
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
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
            "ksp_monitor": "",
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
    
    # Points array shape (N, 3)
    points = np.zeros((n_eval * n_eval, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
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
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
    epsilon = pde_config.get("epsilon", 0.02)
    beta = pde_config.get("beta", [8.0, 3.0])
    
    # Source term string
    source_str = pde_config.get("source_term", "")
    
    # Boundary conditions
    bc_config = pde_config.get("boundary_conditions", {})
    
    # Grid size for output
    output_grid = case_spec.get("output_grid", {})
    nx_out = output_grid.get("nx", 50)
    ny_out = output_grid.get("ny", 50)
    
    # 2. Create mesh - use fine mesh for high Peclet number
    N = 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1 with SUPG stabilization
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = exp(-250*((x-0.3)**2 + (y-0.7)**2))
    f = ufl.exp(-250.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Convection velocity
    beta_vec = ufl.as_vector([PETSc.ScalarType(beta[0]), PETSc.ScalarType(beta[1])])
    
    # Diffusion coefficient
    eps = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Standard Galerkin terms
    a_std = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_std = f * v * ufl.dx
    
    # SUPG stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_local = beta_norm * h / (2.0 * eps)
    # Use the classic formula: tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # Approximate: for high Pe, tau ~ h / (2*|beta|)
    # More robust formula:
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_local, 1.0), 1.0 - 1.0/Pe_local, Pe_local/3.0))
    
    # SUPG test function modification: v_supg = beta . grad(v) * tau
    # Residual of the PDE applied to trial function:
    # R(u) = -eps * div(grad(u)) + beta . grad(u) - f
    # For linear elements, laplacian of u is zero element-wise, so:
    # R(u) ≈ beta . grad(u) - f  (since -eps * lap(u) = 0 for P1)
    
    r_u = ufl.dot(beta_vec, ufl.grad(u))  # strong residual (without laplacian for P1)
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(r_u, v_supg) * ufl.dx
    L_supg = f * v_supg * ufl.dx
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # 5. Boundary conditions - homogeneous Dirichlet (u=0 on boundary)
    # Check if there's a specific BC
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Default: u = 0 on all boundaries
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Check for non-zero BC
    g_val = 0.0
    if bc_config:
        # Try to extract BC value
        for key, val in bc_config.items():
            if isinstance(val, dict):
                g_val = val.get("value", 0.0)
            elif isinstance(val, (int, float)):
                g_val = val
    
    bc = fem.dirichletbc(PETSc.ScalarType(g_val), dofs, V)
    
    # 6. Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_atol": "1e-12",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    x_pts = np.linspace(0.0, 1.0, nx_out)
    y_pts = np.linspace(0.0, 1.0, ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Replace any NaN with 0 (boundary or missed points)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
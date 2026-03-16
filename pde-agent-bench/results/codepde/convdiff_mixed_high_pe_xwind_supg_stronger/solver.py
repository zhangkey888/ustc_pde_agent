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
    beta_vec = pde_config.get("beta", [20.0, 0.0])
    
    # High Pe number -> need fine mesh + SUPG
    # For Pe ~ 4000, we need good stabilization
    N = 128
    degree = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution and source term
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    grad_u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    ])
    laplacian_u_exact = -2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    
    f_expr = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # Standard Galerkin terms
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # Stabilization parameter (stronger SUPG)
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Classical SUPG tau with coth formula approximation
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Use a simpler robust formula:
    tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    # Residual applied to trial function: R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # So residual simplifies to: beta.grad(u) - f
    R_u = ufl.dot(beta, ufl.grad(u)) - f_expr
    
    # SUPG test function modification
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = supg_test * ufl.dot(beta, ufl.grad(u)) * ufl.dx
    L_supg = supg_test * f_expr * ufl.dx
    
    # Also add crosswind diffusion for additional stability
    # Crosswind diffusion: add diffusion in the direction perpendicular to beta
    # delta_cw * h * |residual| * grad(u) . grad(v) in crosswind direction
    # For simplicity, add isotropic artificial diffusion (small amount)
    cw_factor = fem.Constant(domain, PETSc.ScalarType(0.1))
    a_cw = cw_factor * h * beta_norm * 0.5 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx * 0.0  # disabled for now
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # 5. Boundary conditions
    # u_exact = sin(pi*x)*sin(pi*y) = 0 on all boundaries of unit square
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
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
            "ksp_atol": "1e-12",
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
    
    # 7. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.01)
    beta_val = pde_config.get("beta", [0.0, 15.0])
    
    # High Pe number -> need SUPG stabilization and fine mesh
    # Pe ~ |beta| * h / (2*eps), we want reasonable resolution
    N = 128  # mesh resolution
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
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # So -eps * laplacian = eps * 2 * pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u_exact) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    pi_ = ufl.pi
    f_expr = (eps_val * 2.0 * pi_**2 * ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1])
              + beta_val[0] * pi_ * ufl.cos(pi_ * x[0]) * ufl.sin(pi_ * x[1])
              + beta_val[1] * pi_ * ufl.sin(pi_ * x[0]) * ufl.cos(pi_ * x[1]))
    
    # Diffusion coefficient and velocity
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    beta = fem.Constant(domain, np.array(beta_val, dtype=default_scalar_type))
    
    # Standard Galerkin terms
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter (standard formula)
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    # Optimal tau with coth formula approximation
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Use a simpler robust formula:
    tau = h / (2.0 * beta_mag + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    # SUPG: residual applied to test function modification
    # R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 within each cell
    # So residual simplifies to: beta . grad(u) - f
    R_u = ufl.dot(beta, ufl.grad(u)) - f_expr
    
    # SUPG test function: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(R_u, v_supg) * ufl.dx
    # Split into bilinear and linear parts
    # R_u = beta.grad(u) - f
    # a_supg_bilinear = tau * (beta.grad(u)) * (beta.grad(v)) dx
    # L_supg = tau * f * (beta.grad(v)) dx
    
    a_supg_bilinear = tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_supg_linear = tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    
    a = a_std + a_supg_bilinear
    L = L_std + L_supg_linear
    
    # 5. Boundary conditions (u = sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_: np.zeros_like(x_[0]))
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
            "ksp_max_it": "5000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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
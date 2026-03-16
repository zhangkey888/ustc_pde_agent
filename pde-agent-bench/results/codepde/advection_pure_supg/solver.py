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
    
    epsilon = float(params.get("epsilon", 0.0))
    beta = params.get("beta", [10.0, 4.0])
    beta_x = float(beta[0])
    beta_y = float(beta[1])
    
    nx_out = case_spec.get("nx", 50)
    ny_out = case_spec.get("ny", 50)
    
    # For pure advection (epsilon=0), we need SUPG stabilization
    # Use a fine mesh for accuracy
    N = 128
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -epsilon * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So: -epsilon * (-2*pi^2*sin(pi*x)*sin(pi*y)) + beta_x*pi*cos(pi*x)*sin(pi*y) + beta_y*sin(pi*x)*pi*cos(pi*y)
    
    f_ufl = (epsilon * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
             + beta_x * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
             + beta_y * ufl.sin(ufl.pi * x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1]))
    
    # Beta as UFL vector
    beta_ufl = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta_x)),
                               fem.Constant(domain, default_scalar_type(beta_y))])
    
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_gal = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
    L_gal = ufl.inner(f_ufl, v) * ufl.dx
    
    # SUPG stabilization
    # Stabilization parameter tau
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    
    # Compute element Peclet number
    # For epsilon = 0, we use tau = h / (2 * |beta|)
    # General formula with coth-based optimal tau:
    # Pe_h = |beta| * h / (2 * epsilon)
    # tau = h / (2 * |beta|) * (coth(Pe_h) - 1/Pe_h)
    # For epsilon -> 0: tau -> h / (2 * |beta|)
    
    # Use a robust formula
    if epsilon < 1e-12:
        tau = h / (2.0 * beta_norm + 1e-16)
    else:
        Pe_h = beta_norm * h / (2.0 * eps_const)
        # Approximate coth(Pe) - 1/Pe for stability
        # For large Pe: coth(Pe) ~ 1, so result ~ 1 - 1/Pe ~ 1
        # For small Pe: coth(Pe) - 1/Pe ~ Pe/3
        # Use: xi = min(1, Pe/3) as a simpler approximation
        # Or just use h/(2|beta|) which is fine for high Pe
        tau = h / (2.0 * beta_norm + 1e-16)
    
    # SUPG: add stabilization term
    # Residual of the strong form applied to trial function:
    # R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 within each element
    # So R(u) = beta . grad(u) - f
    
    # SUPG test function modification: v_supg = tau * (beta . grad(v))
    v_supg = tau * ufl.dot(beta_ufl, ufl.grad(v))
    
    # For P1 elements, laplacian of u is zero element-wise
    # Strong residual operator on u: beta . grad(u) (the -eps*laplacian part vanishes for P1)
    a_supg = ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v_supg) * ufl.dx
    if epsilon > 1e-12:
        a_supg += eps_const * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) * ufl.dx  # This is not standard for SUPG with P1
    
    L_supg = ufl.inner(f_ufl, v_supg) * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # Boundary conditions: u = g = sin(pi*x)*sin(pi*y) on boundary
    # For the manufactured solution, u = 0 on all boundaries of [0,1]^2
    # since sin(0) = sin(pi) = 0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
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
        petsc_options_prefix="cdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract solution on uniform grid
    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
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
        }
    }
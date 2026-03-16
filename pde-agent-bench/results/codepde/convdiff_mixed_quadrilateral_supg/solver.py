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
    beta_val = pde_config.get("beta", [14.0, 6.0])
    
    # High Peclet number -> need SUPG stabilization and fine mesh
    # Pe ~ |beta| * h / (2*eps) -> need good resolution
    
    nx = ny = 128
    degree = 1
    
    # 2. Create mesh - use triangles (quadrilateral mentioned but triangles work fine for SUPG)
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Manufactured solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Compute source: f = -eps * laplacian(u) + beta . grad(u)
    # u = sin(pi*x)*sin(pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # laplacian(u) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So f = -eps*(-2*pi^2*sin(pi*x)*sin(pi*y)) + beta_x*pi*cos(pi*x)*sin(pi*y) + beta_y*pi*sin(pi*x)*cos(pi*y)
    #       = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta_x*pi*cos(pi*x)*sin(pi*y) + beta_y*pi*sin(pi*x)*cos(pi*y)
    
    beta = ufl.as_vector([default_scalar_type(beta_val[0]), default_scalar_type(beta_val[1])])
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    f_expr = -eps_c * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, grad_u_exact)
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # Peclet number (local)
    Pe_local = beta_norm * h / (2.0 * eps_c)
    
    # SUPG parameter tau
    # Use the standard formula with coth
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2|beta|) * (1 - 1/Pe)
    # Simpler: tau = h / (2 * |beta|) for high Pe
    # More robust formula:
    tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_local, 1.0), 1.0 - 1.0/Pe_local, Pe_local/3.0))
    
    # SUPG residual: L_operator(u) - f applied to trial function
    # L_operator(u) = -eps*laplacian(u) + beta.grad(u)
    # For linear elements, laplacian(u) = 0 within elements
    # So SUPG test modification: v_supg = tau * beta . grad(v)
    
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    # For P1 elements, -eps*laplacian(u_h) = 0 element-wise
    # So the SUPG terms become:
    a_supg = ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # 6. Boundary conditions (u = sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Solve
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
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
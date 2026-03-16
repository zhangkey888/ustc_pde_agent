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
    
    epsilon = float(params.get("epsilon", 0.01))
    beta = params.get("beta", [10.0, 4.0])
    beta_x = float(beta[0])
    beta_y = float(beta[1])
    
    # High Peclet number => need SUPG stabilization and fine mesh
    # Use P2 elements as indicated by case name
    N = 128  # mesh resolution
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = sin(pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Velocity vector
    beta_vec = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta_x)),
                               fem.Constant(domain, default_scalar_type(beta_y))])
    
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian of sin(pi*x)*sin(2*pi*y) = -(pi^2 + 4*pi^2)*sin(pi*x)*sin(2*pi*y) = -5*pi^2*u
    # grad(u_exact) = (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # f = -eps * (-5*pi^2 * u_exact) + beta . grad(u_exact)
    #   = 5*eps*pi^2*u_exact + beta_x*pi*cos(pi*x)*sin(2*pi*y) + beta_y*2*pi*sin(pi*x)*cos(2*pi*y)
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    f_expr = -eps_const * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_standard = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
                 ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    
    # Peclet number based on element size
    beta_mag = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_h = beta_mag * h / (2.0 * eps_const)
    
    # SUPG stabilization parameter
    # tau = h / (2 * |beta|) * (coth(Pe_h) - 1/Pe_h)
    # For high Pe, coth(Pe) - 1/Pe ≈ 1, so tau ≈ h / (2*|beta|)
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_mag) * ufl.min_value(Pe_h / 3.0, ufl.as_ufl(1.0))
    
    # SUPG residual: R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For trial function (linear), laplacian can be computed but for P2 on triangles
    # the second derivatives are piecewise constant
    # Residual applied to trial function:
    # R(u) = -eps * div(grad(u)) + beta . grad(u) - f
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    # For the bilinear form, the SUPG term with the full residual operator:
    # Note: for P2 elements on triangles, div(grad(u)) is piecewise constant (not zero)
    a_supg = (-eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))) * v_supg * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # 6. Boundary conditions - u = g = sin(pi*x)*sin(2*pi*y) on boundary
    # On the unit square boundary, sin(pi*x)*sin(2*pi*y) = 0 everywhere
    # because at x=0,1: sin(pi*0)=sin(pi*1)=0 and at y=0,1: sin(2*pi*0)=sin(2*pi*1)=0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value is 0 on entire boundary for this manufactured solution
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
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
    nx_out = 50
    ny_out = 50
    
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
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
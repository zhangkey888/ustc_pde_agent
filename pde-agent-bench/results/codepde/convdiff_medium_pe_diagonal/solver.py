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
    epsilon = params.get("epsilon", 0.05)
    beta = params.get("beta", [3.0, 3.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # 2. Create mesh - use higher resolution for high Peclet number
    N = 128
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1 with SUPG stabilization
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for manufactured source term
    u_exact_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Velocity vector
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Source term: f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    # u_exact = sin(2*pi*x)*sin(pi*y)
    # laplacian = -(4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y) = -5*pi^2*sin(2*pi*x)*sin(pi*y)
    # -epsilon * laplacian = epsilon * 5 * pi^2 * sin(2*pi*x)*sin(pi*y)
    # grad(u_exact) = (2*pi*cos(2*pi*x)*sin(pi*y), pi*sin(2*pi*x)*cos(pi*y))
    # beta . grad = beta[0]*2*pi*cos(2*pi*x)*sin(pi*y) + beta[1]*pi*sin(2*pi*x)*cos(pi*y)
    
    f_expr = (epsilon * default_scalar_type(5.0) * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + default_scalar_type(beta[0]) * 2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + default_scalar_type(beta[1]) * ufl.pi * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Standard Galerkin terms
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # Stabilization parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG test function modification: tau * (beta . grad(v))
    # Residual applied to trial: -epsilon * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # So residual ≈ beta . grad(u) - f
    r_u = ufl.dot(beta_vec, ufl.grad(u)) - f_expr
    supg_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(r_u, supg_test) * ufl.dx
    # Split into bilinear and linear parts
    a_supg_bilinear = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f_expr, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    
    a = a_standard + a_supg_bilinear
    L = L_standard + L_supg
    
    # 5. Boundary conditions
    # u_exact = sin(2*pi*x)*sin(pi*y) = 0 on all boundaries of [0,1]^2
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
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
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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
            "iterations": iterations,
            "stabilization": "SUPG",
        }
    }
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
    epsilon = params.get("epsilon", 0.01)
    beta = params.get("beta", [10.0, 10.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # 2. Create mesh - use higher resolution for high Peclet number
    N = 128
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Velocity vector
    beta_vec = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Source term from manufactured solution
    # -eps * laplacian(u_exact) + beta . grad(u_exact) = f
    # laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So: -eps * (-2*pi^2*sin(pi*x)*sin(pi*y)) + beta . grad(sin(pi*x)*sin(pi*y))
    # = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    f_expr = (2.0 * epsilon * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Standard Galerkin terms
    a_standard = (eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Use a simpler robust formula
    tau = h / (2.0 * beta_norm + 1e-10)
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    # Residual applied to trial function: -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian of u vanishes element-wise, but for P2 it doesn't
    # However, for SUPG with trial functions, we use the strong form residual
    # R(u) = -eps * div(grad(u)) + beta . grad(u) - f
    # SUPG adds: integral(tau * (beta . grad(v)) * R(u)) dx
    # Since u is a trial function, we split into bilinear and linear parts
    
    # Bilinear SUPG contribution: tau * (beta.grad(v)) * (-eps*laplacian(u) + beta.grad(u))
    # For P2, laplacian doesn't vanish but is constant per cell for triangles
    # We'll skip the diffusion part of the residual in SUPG (common approximation)
    a_supg = ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions
    # u = sin(pi*x)*sin(pi*y) = 0 on all boundaries of [0,1]^2
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
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
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
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
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
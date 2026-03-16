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
    
    epsilon = params.get("epsilon", 0.02)
    beta = params.get("beta", [6.0, 2.0])
    
    # For high Peclet number, use SUPG stabilization and fine mesh
    # Pe ~ 316, so we need stabilization and good resolution
    N = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(2*pi*x)*sin(2*pi*y) = -8*pi^2 * sin(2*pi*x)*sin(2*pi*y)
    # So -eps * (-8*pi^2 * u) = 8*eps*pi^2 * u
    # grad(u) = (2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y))
    # beta . grad(u) = beta[0]*2*pi*cos(2*pi*x)*sin(2*pi*y) + beta[1]*2*pi*sin(2*pi*x)*cos(2*pi*y)
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    f_expr = -epsilon * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 4. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a_std = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) 
             + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # SUPG stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # Use the standard formula with coth-like expression
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # Approximate: for large Pe, tau ~ h/(2*|beta|)
    tau = h / (2.0 * beta_norm + 1e-10) * (1.0 - 1.0 / (Pe_cell + 1e-10))
    # Clamp tau to be non-negative effectively
    # Alternative simpler formula:
    tau = h * h / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG residual: R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # For degree 2, we keep it
    R_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f_expr
    
    # SUPG test function modification
    supg_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = a_std + (-epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))) * supg_test * ufl.dx
    L_supg = L_std + f_expr * supg_test * ufl.dx
    
    # 5. Boundary conditions
    # u = sin(2*pi*x)*sin(2*pi*y) = 0 on boundary of unit square
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
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
    
    # 7. Extract on 50x50 grid
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
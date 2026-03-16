import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.05)
    beta_val = pde_config.get("beta", [4.0, 2.0])
    
    # High Peclet number (~89.4) => use SUPG stabilization
    # Manufactured solution: u = sin(2*pi*x)*sin(pi*y)
    
    # 2. Create mesh - use quadrilateral as specified
    nx, ny = 80, 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space - degree 2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Diffusion coefficient
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    
    # Velocity field
    beta = ufl.as_vector([default_scalar_type(beta_val[0]), default_scalar_type(beta_val[1])])
    
    # Source term from manufactured solution:
    # f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(sin(2*pi*x)*sin(pi*y)) = -(4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y) = -5*pi^2 * u_exact
    # So -eps * laplacian = eps * 5 * pi^2 * u_exact
    # grad(u_exact) = (2*pi*cos(2*pi*x)*sin(pi*y), pi*sin(2*pi*x)*cos(pi*y))
    # beta . grad = 4*2*pi*cos(2*pi*x)*sin(pi*y) + 2*pi*sin(2*pi*x)*cos(pi*y)
    
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    f = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # Standard Galerkin terms
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_standard = f * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    
    # Stabilization parameter (optimal choice)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG: residual applied to test function modification
    # R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian of trial function vanishes within elements
    # For quadratic elements, we include it
    # The strong residual operator on trial function:
    # For SUPG test function: v_supg = tau * beta . grad(v)
    
    r_supg = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = r_supg * v_supg * ufl.dx
    L_supg = f * v_supg * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions
    # u = g = sin(2*pi*x)*sin(pi*y) on boundary
    # On the unit square boundary, sin(2*pi*x)*sin(pi*y) = 0 everywhere:
    # x=0: sin(0)=0, x=1: sin(2*pi)=0, y=0: sin(0)=0, y=1: sin(pi)=0
    # So homogeneous Dirichlet BC
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
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
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
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
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    epsilon = pde_config.get("epsilon", 0.05)
    beta = pde_config.get("beta", [4.0, 0.0])
    
    # Grid for output
    nx_out = 50
    ny_out = 50
    
    # Mesh resolution - need fine enough mesh for Pe~80
    # With SUPG we can use moderate resolution
    N = 128
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Spatial coordinate and exact solution
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(2*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -eps * laplacian(u) + beta . grad(u)
    # u = exp(2x)*sin(pi*y)
    # du/dx = 2*exp(2x)*sin(pi*y)
    # du/dy = pi*exp(2x)*cos(pi*y)
    # d2u/dx2 = 4*exp(2x)*sin(pi*y)
    # d2u/dy2 = -pi^2*exp(2x)*sin(pi*y)
    # laplacian = (4 - pi^2)*exp(2x)*sin(pi*y)
    # f = -eps*(4 - pi^2)*exp(2x)*sin(pi*y) + beta[0]*2*exp(2x)*sin(pi*y) + beta[1]*pi*exp(2x)*cos(pi*y)
    
    f_expr = (-epsilon * (4.0 - ufl.pi**2) * ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * 2.0 * ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.pi * ufl.exp(2.0 * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    beta_vec = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Standard Galerkin
    a_standard = (eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe), approximated
    # Use a simpler robust formula:
    tau = h / (2.0 * beta_norm + 1e-10) * (1.0 - 1.0 / (Pe_cell + 1e-10))
    # Clamp tau to be non-negative via min with simpler expression
    # Actually use the standard formula:
    tau = h**2 / (4.0 * eps_const + 2.0 * beta_norm * h)
    
    # SUPG test function modification: v_supg = beta . grad(v)
    r_u = -eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    v_supg = ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = a_standard + tau * ufl.inner(r_u, v_supg) * ufl.dx
    L_supg = L_standard + tau * f_expr * v_supg * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Exact solution for BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x_arr: np.exp(2.0 * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # 7. Solve
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
    
    # 8. Extract solution on uniform grid
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
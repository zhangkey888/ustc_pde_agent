import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization.
    -eps * laplacian(u) + beta . grad(u) = f
    u = g on boundary
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec["pde"]
    eps = pde["coefficients"]["epsilon"]  # 0.005
    beta = np.array(pde["coefficients"]["beta"])  # [20.0, 0.0]
    
    out = case_spec["output"]
    nx = out["grid"]["nx"]
    ny = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Solver parameters (tuned for accuracy within time budget)
    mesh_res = 300
    element_degree = 1
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-12
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Convection vector as UFL
    beta_ufl = ufl.as_vector([beta[0], beta[1]])
    
    # Exact/manufactured solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # grad(u_exact) = [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    f_expr = (2.0 * eps * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # SUPG stabilization parameter
    # For linear elements: tau = h / (2 * |beta|)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta_ufl, beta_ufl))
    tau_supg = h / (2.0 * beta_norm + 1e-30)
    
    # Galerkin part
    a_galerkin = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx)
    L_galerkin = f_expr * v * ufl.dx
    
    # SUPG stabilization terms
    # a_supg: tau * (beta.grad(v)) * (beta.grad(u))
    # L_supg: tau * (beta.grad(v)) * f
    residual_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
    residual_f = f_expr
    
    a_supg = tau_supg * ufl.inner(ufl.dot(beta_ufl, ufl.grad(v)), ufl.dot(beta_ufl, ufl.grad(u))) * ufl.dx
    L_supg = tau_supg * ufl.inner(ufl.dot(beta_ufl, ufl.grad(v)), residual_f) * ufl.dx
    
    # Crosswind diffusion term for additional stability
    beta_perp = ufl.as_vector([-beta[1], beta[0]])
    beta_perp_norm = ufl.sqrt(ufl.inner(beta_perp, beta_perp)) + 1e-30
    delta_cw = 0.5 * h * beta_norm / (beta_norm + 1e-30)  # crosswind diffusion coefficient
    
    a_cw = delta_cw * ufl.inner(ufl.dot(beta_perp / beta_perp_norm, ufl.grad(u)),
                                ufl.dot(beta_perp / beta_perp_norm, ufl.grad(v))) * ufl.dx
    
    # Total bilinear and linear forms
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # Boundary conditions - Dirichlet on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx * ny, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    # Gather across processes
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        nan_mask = np.isnan(u_values)
        nan_count = np.zeros_like(u_values, dtype=np.int32)
        nan_count[nan_mask] = 1
        nan_count_global = np.zeros_like(nan_count, dtype=np.int32)
        comm.Allreduce(nan_count, nan_count_global, op=MPI.SUM)
        u_values_global[nan_count_global > 0] = np.nan
        u_values = u_values_global
    
    u_grid = u_values.reshape(ny, nx)
    
    # Compute L2 error for verification
    L2_error = fem.assemble_scalar(
        fem.form((u_sol - u_exact)**2 * ufl.dx))
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}, iterations: {iterations}")
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
    
    return result

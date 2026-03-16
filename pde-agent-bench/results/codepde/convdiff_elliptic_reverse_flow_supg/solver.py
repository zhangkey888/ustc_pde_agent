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
    beta = params.get("beta", [-8.0, 4.0])
    
    # High Peclet number (~447) => need SUPG stabilization and fine mesh
    N = 128
    degree = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(x)*sin(pi*y)
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    # grad(u_exact) = (exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    # laplacian(u_exact) = exp(x)*sin(pi*y) - pi^2*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)*(1-pi^2)
    # f = -epsilon*(1-pi^2)*exp(x)*sin(pi*y) + beta[0]*exp(x)*sin(pi*y) + beta[1]*exp(x)*pi*cos(pi*y)
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    f_expr = -eps_const * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 4. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    h = ufl.CellDiameter(domain)
    
    # Standard Galerkin terms
    a_standard = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
                 ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau_supg with coth formula approximation
    # Use the standard formula: tau = h/(2*|beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Simpler: tau = h / (2 * |beta|) * min(1, Pe/3)
    # Or use the "optimal" formula for linear elements
    tau_supg = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 
                                                          1.0 - 1.0/Pe_cell, 
                                                          Pe_cell/3.0))
    
    # Residual applied to trial function: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within each element
    # So residual_trial = beta.grad(u) - f (since -eps*0 + beta.grad(u))
    residual_u = ufl.dot(beta_vec, ufl.grad(u))
    
    # SUPG test function modification
    supg_weight = tau_supg * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = supg_weight * residual_u * ufl.dx
    L_supg = supg_weight * f_expr * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
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
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract on 50x50 uniform grid
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
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
    epsilon = params.get("epsilon", 0.005)
    beta = params.get("beta", [20.0, 10.0])
    
    # Grid for output
    nx_out = 50
    ny_out = 50
    
    # Mesh resolution - need fine mesh for high Peclet number
    N = 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Velocity field
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Source term from manufactured solution:
    # f = -epsilon * laplacian(u) + beta . grad(u)
    # u = sin(pi*x)*sin(pi*y)
    # laplacian(u) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u) = [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    # f = epsilon * 2*pi^2 * sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    f_expr = (epsilon * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # 4. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard choice)
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # Use the formula: tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Simpler robust formula:
    tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                             1.0 - 1.0 / Pe_cell,
                                                             Pe_cell / 3.0))
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    # Residual applied to trial: -epsilon * laplacian(u) + beta . grad(u)
    # For linear elements, laplacian(u) = 0 within elements
    # For quadratic elements, we include it
    
    # Strong residual of the operator applied to u (trial):
    # R(u) = -epsilon * div(grad(u)) + beta . grad(u)
    # SUPG: add integral of tau * (beta . grad(v)) * R(u) dx
    # For the bilinear form:
    # a_supg = tau * (beta . grad(v)) * (-epsilon * laplacian(u) + beta . grad(u)) dx
    # For the linear form:
    # L_supg = tau * (beta . grad(v)) * f dx
    
    beta_grad_v = ufl.dot(beta_vec, ufl.grad(v))
    beta_grad_u = ufl.dot(beta_vec, ufl.grad(u))
    
    # For P2 elements, laplacian is nonzero but piecewise constant
    # -epsilon * div(grad(u)) term in SUPG
    # We include it for better accuracy
    a_supg = tau * beta_grad_v * (-epsilon * ufl.div(ufl.grad(u)) + beta_grad_u) * ufl.dx
    L_supg = tau * beta_grad_v * f_expr * ufl.dx
    
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # 5. Boundary conditions
    # u = sin(pi*x)*sin(pi*y) = 0 on all boundaries of [0,1]^2
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_gmres_restart": "100",
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
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = xx.ravel()
    points[1] = yy.ravel()
    
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
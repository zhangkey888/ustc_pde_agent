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
    epsilon = params.get("epsilon", 0.03)
    beta = params.get("beta", [5.0, 2.0])
    
    # Grid for output
    output_grid = case_spec.get("output_grid", {})
    nx_out = output_grid.get("nx", 50)
    ny_out = output_grid.get("ny", 50)
    
    # High Peclet number ~179.5, need SUPG stabilization and fine mesh with P2
    mesh_resolution = 80
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, mesh.CellType.triangle)
    
    # 3. Function space - P2 for better accuracy
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: sin(pi*x)*sin(2*pi*y)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Velocity field
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Source term from manufactured solution:
    # f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(sin(pi*x)*sin(2*pi*y)) = -(pi^2 + 4*pi^2)*sin(pi*x)*sin(2*pi*y) = -5*pi^2*sin(pi*x)*sin(2*pi*y)
    # So -epsilon * laplacian = epsilon * 5*pi^2 * sin(pi*x)*sin(2*pi*y)
    # grad(u_exact) = (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # beta . grad = 5*pi*cos(pi*x)*sin(2*pi*y) + 2*2*pi*sin(pi*x)*cos(2*pi*y)
    
    f_expr = (epsilon * 5.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
              + beta[1] * 2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]))
    
    # Standard Galerkin terms
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2*|beta|) for large Pe
    # Simpler formula that works well:
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp: use min with a simpler approach
    # For high Pe, tau ~ h/(2*|beta|)
    # Use the standard formula:
    tau = h * h / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG: add stabilization term
    # Residual of strong form applied to trial function:
    # R(u) = -epsilon * div(grad(u)) + beta . grad(u) - f
    # For linear elements, div(grad(u)) = 0 within elements, but for P2 it's nonzero
    residual_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    residual_f = f_expr
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(residual_u, v_supg) * ufl.dx
    L_supg = ufl.inner(residual_f, v_supg) * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 5. Boundary conditions - u = 0 on all boundaries (sin(pi*x)*sin(2*pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
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
            "ksp_gmres_restart": "100",
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
    points[0] = X.ravel()
    points[1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
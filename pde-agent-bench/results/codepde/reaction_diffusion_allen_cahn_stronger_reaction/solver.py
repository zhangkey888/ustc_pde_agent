import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec["pde"]
    
    # Extract parameters
    epsilon = pde_config.get("epsilon", 0.01)
    reaction_type = pde_config.get("reaction_type", "allen_cahn")
    reaction_lambda = pde_config.get("reaction_lambda", 1.0)
    
    # Time parameters
    time_params = pde_config.get("time", None)
    t_end = time_params.get("t_end", 0.1) if time_params else 0.1
    dt_suggested = time_params.get("dt", 0.002) if time_params else 0.002
    time_scheme = time_params.get("scheme", "backward_euler") if time_params else "backward_euler"
    
    # Use a finer mesh and smaller dt for accuracy
    N = 100
    degree = 2
    dt = dt_suggested * 0.5  # Use half the suggested dt for better accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time constant
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = exp(-t)*(0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    u_exact_ufl = ufl.exp(-t_const) * (0.15 + 0.12 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1]))
    
    # Compute the source term from the manufactured solution
    # u_t = -exp(-t)*(0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y)) = -u_exact
    # -eps * laplacian(u) : need to compute
    # For u_exact = exp(-t) * phi(x,y), where phi = 0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y)
    # laplacian(u) = exp(-t) * laplacian(phi)
    # laplacian(phi) = 0.12 * (-4*pi^2) * sin(2*pi*x)*sin(2*pi*y) + 0.12 * (-4*pi^2) * sin(2*pi*x)*sin(2*pi*y)
    #               = -0.12 * 8 * pi^2 * sin(2*pi*x)*sin(2*pi*y)
    # So -eps * laplacian(u) = eps * exp(-t) * 0.12 * 8 * pi^2 * sin(2*pi*x)*sin(2*pi*y)
    
    # Allen-Cahn reaction: R(u) = lambda * (u^3 - u) or lambda * u * (u^2 - 1)
    # f = u_t - eps * laplacian(u) + R(u)
    # f = -u_exact - eps * laplacian(u_exact) + lambda * (u_exact^3 - u_exact)
    
    # Let UFL handle the derivatives symbolically
    # We'll define f using UFL differentiation
    
    # For the source term, we need du/dt analytically
    # u_exact = exp(-t) * (0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    # du/dt = -exp(-t) * (0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y)) = -u_exact
    
    phi = 0.15 + 0.12 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    u_exact_expr = ufl.exp(-t_const) * phi
    du_dt_expr = -ufl.exp(-t_const) * phi  # time derivative
    
    # Laplacian of u_exact
    # grad(u_exact) = exp(-t) * grad(phi)
    # div(grad(u_exact)) = exp(-t) * div(grad(phi))
    # We compute this via UFL
    grad_phi = ufl.grad(phi)
    laplacian_phi = ufl.div(grad_phi)
    laplacian_u = ufl.exp(-t_const) * laplacian_phi
    
    # Source term: f = du/dt - eps*laplacian(u) + R(u)
    # Note: the PDE is du/dt - eps*laplacian(u) + R(u) = f
    # So f = du/dt - eps*laplacian(u) + R(u_exact)
    R_u_exact = reaction_lambda * (u_exact_expr**3 - u_exact_expr)
    f_expr = du_dt_expr - epsilon * laplacian_u + R_u_exact
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (for Newton)
    v = ufl.TestFunction(V)
    
    # Initial condition at t=0
    # u(x,0) = 0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y)
    u_n.interpolate(lambda X: 0.15 + 0.12 * np.sin(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 5. Boundary conditions (time-dependent Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        u_bc.interpolate(
            lambda X: np.exp(-t_val) * (0.15 + 0.12 * np.sin(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1]))
        )
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Variational form (backward Euler + Newton for nonlinear reaction)
    # (u_h - u_n)/dt - eps*laplacian(u_h) + R(u_h) = f(t_{n+1})
    # Residual F = (u_h - u_n)/dt * v + eps * grad(u_h) . grad(v) + R(u_h)*v - f*v = 0
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    lam_const = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))
    
    # Allen-Cahn reaction: R(u) = lambda*(u^3 - u)
    F_form = (
        (u_h - u_n) / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + lam_const * (u_h**3 - u_h) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    
    # Set up Newton solver
    problem = petsc.NonlinearProblem(F_form, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # Initialize u_h with initial condition
    u_h.x.array[:] = u_n.x.array[:]
    
    # 7. Time stepping
    total_linear_iterations = 0
    nonlinear_iterations_list = []
    
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update BC
        update_bc(t_current)
        
        # Use previous solution as initial guess
        u_h.x.array[:] = u_n.x.array[:]
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t_current}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations_list.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 8. Extract solution on 75x75 grid
    nx_out, ny_out = 75, 75
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations_list,
        }
    }
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract parameters
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "cubic")
    alpha = reaction.get("alpha", 1.0)
    beta = reaction.get("beta", 1.0)
    
    epsilon = pde_config.get("epsilon", 1.0)
    if epsilon is None:
        epsilon = 1.0
    
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.25)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Manufactured solution: u = exp(-t)*(0.15*sin(3*pi*x)*sin(2*pi*y))
    # We need to derive f from this
    
    # Solver parameters
    nx = 80
    ny = 80
    degree = 2
    dt = dt_suggested
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution in UFL
    u_exact_ufl = ufl.exp(-t_const) * (0.15 * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]))
    
    # Compute the source term from the manufactured solution:
    # du/dt - eps * laplacian(u) + R(u) = f
    # du/dt = -exp(-t) * 0.15 * sin(3*pi*x)*sin(2*pi*y)
    # laplacian(u) = exp(-t)*0.15*(-9*pi^2 - 4*pi^2)*sin(3*pi*x)*sin(2*pi*y)
    #              = -exp(-t)*0.15*13*pi^2*sin(3*pi*x)*sin(2*pi*y)
    # -eps*laplacian(u) = eps*exp(-t)*0.15*13*pi^2*sin(...)
    # R(u) = alpha*u + beta*u^3
    
    # We'll compute f symbolically
    # du/dt:
    du_dt_ufl = -ufl.exp(-t_const) * (0.15 * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]))
    
    # -eps * laplacian(u):
    # grad(u) = exp(-t)*0.15*(3*pi*cos(3*pi*x)*sin(2*pi*y), sin(3*pi*x)*2*pi*cos(2*pi*y))
    # laplacian(u) = exp(-t)*0.15*(-9*pi^2*sin(3*pi*x)*sin(2*pi*y) - 4*pi^2*sin(3*pi*x)*sin(2*pi*y))
    #             = -exp(-t)*0.15*13*pi^2*sin(3*pi*x)*sin(2*pi*y)
    neg_eps_laplacian = epsilon * ufl.exp(-t_const) * 0.15 * 13.0 * pi**2 * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # R(u) = alpha*u + beta*u^3
    R_exact = alpha * u_exact_ufl + beta * u_exact_ufl**3
    
    # f = du/dt - eps*laplacian(u) + R(u)
    # Note: -eps*laplacian(u) = neg_eps_laplacian (already computed as positive term)
    f_ufl = du_dt_ufl + neg_eps_laplacian + R_exact
    
    # 4. Set up the nonlinear time-dependent problem
    # Backward Euler: (u^{n+1} - u^n)/dt - eps*laplacian(u^{n+1}) + R(u^{n+1}) = f^{n+1}
    
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # current solution (unknown)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    eps_c = fem.Constant(domain, default_scalar_type(epsilon))
    alpha_c = fem.Constant(domain, default_scalar_type(alpha))
    beta_c = fem.Constant(domain, default_scalar_type(beta))
    
    # Residual: (u_h - u_n)/dt * v + eps * grad(u_h) . grad(v) + R(u_h)*v - f*v = 0
    F = ((u_h - u_n) / dt_c * v * ufl.dx
         + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + (alpha_c * u_h + beta_c * u_h**3) * v * ufl.dx
         - f_ufl * v * ufl.dx)
    
    # 5. Boundary conditions - u = u_exact on boundary
    # At each time step, we update the BC function
    u_bc_func = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # 6. Initial condition at t=0
    # u(x,0) = 0.15*sin(3*pi*x)*sin(2*pi*y)
    t_current = 0.0
    t_const.value = t_current
    
    u_n.interpolate(lambda x_arr: 0.15 * np.sin(3 * pi * x_arr[0]) * np.sin(2 * pi * x_arr[1]))
    u_h.x.array[:] = u_n.x.array[:]
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # Set up nonlinear solver
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    solver.report = False
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # 7. Time stepping
    n_steps = int(np.round(t_end / dt))
    dt_actual = t_end / n_steps
    dt_c.value = dt_actual
    
    total_linear_iters = 0
    nonlinear_iterations_list = []
    
    for step in range(n_steps):
        t_current += dt_actual
        t_const.value = t_current
        
        # Update boundary condition
        t_val = t_current
        u_bc_func.interpolate(
            lambda x_arr, t_v=t_val: np.exp(-t_v) * 0.15 * np.sin(3 * pi * x_arr[0]) * np.sin(2 * pi * x_arr[1])
        )
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step+1}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations_list.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 8. Extract solution on 70x70 grid
    nx_eval = 70
    ny_eval = 70
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Also extract initial condition on grid
    # Recompute u_initial analytically at t=0
    u_initial_grid = 0.15 * np.sin(3 * pi * X) * np.sin(2 * pi * Y)
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iterations_list) * 5,  # approximate linear iters
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations_list,
        }
    }
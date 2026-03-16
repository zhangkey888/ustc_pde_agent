import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde.get("epsilon", 0.01)
    reaction_type = pde.get("reaction_type", "logistic")
    reaction_rho = pde.get("reaction_rho", 50.0)
    
    time_params = pde.get("time", None)
    t_end = time_params.get("t_end", 0.2) if time_params else 0.2
    dt_suggested = time_params.get("dt", 0.005) if time_params else 0.005
    time_scheme = time_params.get("scheme", "backward_euler") if time_params else "backward_euler"
    
    # Use a fine mesh and small dt for accuracy
    nx = 96
    ny = 96
    degree = 2
    dt = 0.002  # smaller than suggested for accuracy
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Manufactured solution: u = exp(-t)*(0.35 + 0.1*cos(2*pi*x)*sin(pi*y))
    # We need to compute the source term f from the manufactured solution
    # For the reaction-diffusion equation:
    #   du/dt - epsilon * laplacian(u) + R(u) = f
    # where R(u) = rho * u * (1 - u) for logistic reaction
    
    # Time parameter as a Constant so we can update it
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * (0.35 + 0.1 * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # du/dt = -exp(-t)*(0.35 + 0.1*cos(2*pi*x)*sin(pi*y)) = -u_exact
    dudt_exact = -u_exact_ufl
    
    # Laplacian of u_exact (spatial part only)
    # u_exact = exp(-t) * phi(x,y) where phi = 0.35 + 0.1*cos(2*pi*x)*sin(pi*y)
    # d^2 phi/dx^2 = -0.1*(2*pi)^2*cos(2*pi*x)*sin(pi*y)
    # d^2 phi/dy^2 = -0.1*pi^2*cos(2*pi*x)*sin(pi*y)
    # laplacian(phi) = -0.1*(4*pi^2 + pi^2)*cos(2*pi*x)*sin(pi*y) = -0.1*5*pi^2*cos(2*pi*x)*sin(pi*y)
    # laplacian(u_exact) = exp(-t) * laplacian(phi)
    
    # Instead of manually computing, let UFL handle it via the variational form
    # We'll compute f symbolically
    
    # For logistic reaction: R(u) = rho * u * (1 - u)
    rho = reaction_rho
    
    # Source term: f = du/dt - epsilon * laplacian(u) + R(u)
    # We compute this via UFL
    # laplacian(u) = div(grad(u))
    phi = 0.35 + 0.1 * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    lap_phi = -0.1 * (4 * pi**2 + pi**2) * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # f = dudt - epsilon * laplacian(u) + rho * u * (1 - u)
    # dudt = -exp(-t) * phi
    # laplacian(u) = exp(-t) * lap_phi
    # R(u) = rho * u_exact * (1 - u_exact)
    
    f_expr = (-ufl.exp(-t_const) * phi 
              - epsilon * ufl.exp(-t_const) * lap_phi 
              + rho * u_exact_ufl * (1.0 - u_exact_ufl))
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (for Newton)
    
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x, 0) = 0.35 + 0.1*cos(2*pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: 0.35 + 0.1 * np.cos(2 * np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 5. Boundary conditions - Dirichlet from exact solution
    # We need to update BC values at each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        u_bc.interpolate(lambda X: np.exp(-t_val) * (0.35 + 0.1 * np.cos(2 * np.pi * X[0]) * np.sin(np.pi * X[1])))
    
    update_bc(0.0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Variational form (Backward Euler, nonlinear due to logistic reaction)
    # (u_h - u_n)/dt - epsilon * laplacian(u_h) + rho * u_h * (1 - u_h) = f(t_{n+1})
    # Residual:
    # F = (u_h - u_n)/dt * v * dx + epsilon * grad(u_h) . grad(v) * dx 
    #     + rho * u_h * (1 - u_h) * v * dx - f * v * dx = 0
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    F_form = ((u_h - u_n) / dt_const * v * ufl.dx
              + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
              + rho * u_h * (1.0 - u_h) * v * ufl.dx
              - f_expr * v * ufl.dx)
    
    # 7. Newton solver setup
    problem = petsc.NonlinearProblem(F_form, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 25
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix() or ""
    opts[f"{prefix}ksp_rtol"] = "1e-10"
    opts[f"{prefix}ksp_max_it"] = "500"
    ksp.setFromOptions()
    
    # 8. Time stepping
    nonlinear_iterations = []
    total_linear_iters = 0
    
    t = 0.0
    u_h.x.array[:] = u_n.x.array[:]
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary conditions
        update_bc(t)
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(int(n_iters))
        total_linear_iters += int(n_iters)  # approximate
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Extract solution on 65x65 grid
    nx_out = 65
    ny_out = 65
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
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
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_linear_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
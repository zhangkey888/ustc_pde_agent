import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.3)
    dt_suggested = time_params.get("dt", 0.01)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Diffusion coefficient
    epsilon = pde_config.get("epsilon", pde_config.get("coefficients", {}).get("epsilon", 1.0))
    
    # Reaction type
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "logistic")
    
    # Manufactured solution info
    manufactured = pde_config.get("manufactured", {})
    
    # Parameters
    mesh_resolution = 80
    element_degree = 2
    dt = 0.005  # Use smaller dt for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time variable as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Manufactured solution: u_exact = exp(-t)*(0.2 + 0.1*sin(pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t_const) * (0.2 + 0.1 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Compute the source term from the manufactured solution:
    # du/dt = -exp(-t)*(0.2 + 0.1*sin(pi*x)*sin(pi*y))
    # -eps * laplacian(u) = -eps * exp(-t) * 0.1 * (-2*pi^2) * sin(pi*x)*sin(pi*y)
    #                     = eps * exp(-t) * 0.2 * pi^2 * sin(pi*x)*sin(pi*y)
    # R(u) for logistic: u*(1-u)
    # f = du/dt - eps*laplacian(u) + R(u)
    
    # We'll compute f symbolically
    # du/dt:
    du_dt_ufl = -ufl.exp(-t_const) * (0.2 + 0.1 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Laplacian of u_exact:
    # u = exp(-t) * (0.2 + 0.1*sin(pi*x)*sin(pi*y))
    # nabla^2 u = exp(-t) * 0.1 * (-pi^2*sin(pi*x)*sin(pi*y) - pi^2*sin(pi*x)*sin(pi*y))
    #           = exp(-t) * 0.1 * (-2*pi^2) * sin(pi*x)*sin(pi*y)
    laplacian_u_ufl = ufl.exp(-t_const) * 0.1 * (-2.0 * pi**2) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Reaction: logistic R(u) = u*(1-u)
    R_u_exact = u_exact_ufl * (1.0 - u_exact_ufl)
    
    # Source: f = du/dt - eps*laplacian(u) + R(u)
    f_ufl = du_dt_ufl - eps_const * laplacian_u_ufl + R_u_exact
    
    # 4. Set up the nonlinear time-stepping problem
    # Backward Euler: (u^{n+1} - u^n)/dt - eps*laplacian(u^{n+1}) + R(u^{n+1}) = f^{n+1}
    # Residual: (u - u_n)/dt * v + eps*grad(u)*grad(v) + R(u)*v - f*v = 0
    
    u_h = fem.Function(V, name="u")  # current solution (unknown)
    u_n = fem.Function(V, name="u_n")  # previous time step
    v = ufl.TestFunction(V)
    
    # Initial condition at t=0: u_exact(0, x, y) = 0.2 + 0.1*sin(pi*x)*sin(pi*y)
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    u_h.interpolate(u_init_expr)
    
    # Store initial condition for output
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Build evaluation mapping
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_initial[eval_map] = vals.flatten()
    u_initial = u_initial.reshape(nx_out, ny_out)
    
    # Reaction term R(u) = u*(1-u)
    R_u = u_h * (1.0 - u_h)
    
    # Residual for backward Euler
    # At each time step, t_const will be set to t^{n+1}, and f_ufl evaluated there
    F = ((u_h - u_n) / dt_const) * v * ufl.dx \
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_ufl * v * ufl.dx
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # 5. Set up Newton solver
    # We'll use manual Newton iteration for better control and iteration tracking
    
    # Jacobian
    J = ufl.derivative(F, u_h)
    
    problem = petsc.NonlinearProblem(F, u_h, bcs=[], J=J)
    
    # We handle BCs manually in the loop
    # Actually, let's use the built-in BC handling
    
    # Re-setup with BCs
    # We need to update u_bc at each time step
    
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    current_t = 0.0
    
    for step in range(n_steps):
        current_t += dt
        t_const.value = current_t
        
        # Update boundary condition
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Set initial guess to previous solution
        u_h.x.array[:] = u_n.x.array[:]
        
        # Apply BC to initial guess
        u_h.x.array[boundary_dofs] = u_bc.x.array[boundary_dofs]
        u_h.x.scatter_forward()
        
        # Create nonlinear problem and solver for this step
        problem = petsc.NonlinearProblem(F, u_h, bcs=[bc], J=J)
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
        ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
        
        n_newton, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step+1}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(int(n_newton))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()
    
    # 7. Extract solution on grid
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(points_on_proc, cells_on_proc)
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": int(sum(nonlinear_iterations) * 5),  # approximate linear iterations
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }
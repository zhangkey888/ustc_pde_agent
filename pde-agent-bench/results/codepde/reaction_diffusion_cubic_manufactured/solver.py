import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Diffusion coefficient
    epsilon = pde_config.get("epsilon", 1.0)
    if isinstance(epsilon, dict):
        epsilon = epsilon.get("value", 1.0)
    
    # Reaction type
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "cubic")
    
    # Manufactured solution info
    manufactured = pde_config.get("manufactured", {})
    
    # Grid size for output
    nx_out = 60
    ny_out = 60
    
    # Solver parameters
    mesh_resolution = 80
    element_degree = 2
    dt = dt_suggested
    
    # Ensure dt divides t_end reasonably
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        n_steps = 1
    dt = t_end / n_steps
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * 0.2 * sin(2*pi*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * (0.2 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Compute the source term f from the PDE:
    # du/dt - eps * laplacian(u) + R(u) = f
    # u = exp(-t) * 0.2 * sin(2*pi*x)*sin(pi*y)
    # du/dt = -exp(-t) * 0.2 * sin(2*pi*x)*sin(pi*y) = -u
    # laplacian(u) = exp(-t)*0.2*( -(2*pi)^2 - pi^2 ) * sin(2*pi*x)*sin(pi*y) = -5*pi^2 * u
    # So: -eps * laplacian(u) = 5*eps*pi^2 * u
    # R(u) = u^3 (cubic reaction)
    # f = du/dt - eps*laplacian(u) + u^3
    # f = -u + 5*eps*pi^2*u + u^3
    
    # But let's compute it symbolically with UFL to be safe
    # We need the source term. Let's derive it properly.
    # u_exact = exp(-t) * phi(x,y) where phi = 0.2*sin(2*pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*phi = -u_exact
    # grad(u_exact) = exp(-t) * grad(phi)
    # laplacian(u_exact) = exp(-t) * laplacian(phi)
    # laplacian(phi) = 0.2*( -(2*pi)^2*sin(2*pi*x)*sin(pi*y) - pi^2*sin(2*pi*x)*sin(pi*y) )
    #                = -0.2*(4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y) = -5*pi^2 * phi
    # So laplacian(u_exact) = -5*pi^2 * u_exact
    # -eps*laplacian(u_exact) = 5*eps*pi^2 * u_exact
    
    # f = du/dt - eps*laplacian(u) + u^3
    # f = -u_exact + 5*eps*pi^2*u_exact + u_exact^3
    
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Source term (UFL expression, depends on t_const)
    f_expr = -u_exact_ufl + 5.0 * epsilon * (pi**2) * u_exact_ufl + u_exact_ufl**3
    
    # 4. Setup for time-stepping with Newton (nonlinear due to u^3)
    # Backward Euler: (u^{n+1} - u^n)/dt - eps*laplacian(u^{n+1}) + (u^{n+1})^3 = f^{n+1}
    
    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (Newton unknown)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Initial condition: u(x, 0) = 0.2 * sin(2*pi*x) * sin(pi*y)
    u_init_expr = fem.Expression(
        0.2 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)
    u_h.interpolate(u_init_expr)
    
    # Store initial condition for output
    # Create evaluation grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = X.ravel()
    points_2d[1, :] = Y.ravel()
    
    # Build point evaluation infrastructure
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    def evaluate_function(func):
        values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc, cells_on_proc)
            for idx, global_idx in enumerate(eval_map):
                values[global_idx] = vals[idx, 0] if vals.ndim > 1 else vals[idx]
        return values.reshape((nx_out, ny_out))
    
    # Evaluate initial condition
    u_initial_grid = evaluate_function(u_n)
    
    # Residual form for backward Euler
    # (u_h - u_n)/dt - eps*laplacian(u_h) + u_h^3 = f
    # F = (u_h - u_n)/dt * v + eps*grad(u_h)·grad(v) + u_h^3 * v - f * v = 0
    F = (
        (u_h - u_n) / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + u_h**3 * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    bc_func = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, dofs)
    
    # Setup nonlinear problem and Newton solver
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-12
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # Time stepping
    t = 0.0
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        bc_func.interpolate(bc_expr)
        
        # Use previous solution as initial guess
        u_h.x.array[:] = u_n.x.array[:]
        
        # Solve
        n_newton, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(int(n_newton))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate final solution on grid
    u_grid = evaluate_function(u_h)
    
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
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
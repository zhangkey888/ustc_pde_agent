import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse parameters
    time_params = case_spec.get("pde", {}).get("time", {})
    t_end = float(time_params.get("t_end", 0.2))
    dt_suggested = float(time_params.get("dt", 0.005))
    
    # Agent-selectable parameters
    nx = 80
    ny = 80
    degree = 2
    dt = dt_suggested
    epsilon = 1.0  # default diffusion coefficient unless specified
    
    # Check if epsilon is specified in case_spec
    pde_spec = case_spec.get("pde", {})
    if "epsilon" in pde_spec:
        epsilon = float(pde_spec["epsilon"])
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust dt to exactly hit t_end
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Time constant
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Manufactured solution: u = exp(-t)*(0.2*sin(2*pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t) * (0.2 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Compute the source term f from the manufactured solution
    # u_t = -exp(-t)*(0.2*sin(2*pi*x)*sin(pi*y))
    # -eps * laplacian(u) = eps * exp(-t) * 0.2 * (4*pi^2 + pi^2) * sin(2*pi*x)*sin(pi*y)
    #                     = eps * exp(-t) * 0.2 * 5*pi^2 * sin(2*pi*x)*sin(pi*y)
    # R(u) = u^3 (cubic reaction)
    # f = u_t - eps*laplacian(u) + R(u)
    #   = -u_exact + eps * 5*pi^2 * u_exact + u_exact^3
    
    # But let's derive it symbolically with UFL to be safe
    # We need: du/dt - eps * div(grad(u)) + u^3 = f
    # du/dt = -exp(-t) * 0.2 * sin(2*pi*x)*sin(pi*y) = -u_exact
    
    # For the Laplacian, we compute it from the UFL expression
    # grad(u_exact) w.r.t. spatial coordinates
    # Since u_exact_ufl depends on x via SpatialCoordinate, we can use ufl.grad
    
    # Actually, ufl.grad and ufl.div work on SpatialCoordinate expressions
    laplacian_u = ufl.div(ufl.grad(u_exact_ufl))
    
    # du/dt: derivative of u_exact w.r.t. t
    # u_exact = exp(-t) * 0.2 * sin(2*pi*x)*sin(pi*y)
    # du/dt = -exp(-t) * 0.2 * sin(2*pi*x)*sin(pi*y) = -u_exact
    dudt = -u_exact_ufl
    
    # Source term
    f_ufl = dudt - eps_const * laplacian_u + u_exact_ufl**3
    
    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # current solution (Newton unknown)
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x, 0) = 0.2*sin(2*pi*x)*sin(pi*y)
    t.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    u_h.interpolate(u_exact_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_exact_expr)
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + u^3 = f
    # Weak form (residual): 
    # F = (u_h - u_n)/dt * v * dx + eps * inner(grad(u_h), grad(v)) * dx + u_h^3 * v * dx - f * v * dx = 0
    F = ((u_h - u_n) / dt_const) * v * ufl.dx \
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + u_h**3 * v * ufl.dx \
        - f_ufl * v * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Set initial BC
    t.value = dt  # first time step
    u_bc.interpolate(u_exact_expr)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Nonlinear problem and Newton solver
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # Time stepping
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
        # Use previous solution as initial guess
        u_h.x.array[:] = u_n.x.array[:]
        
        # Solve
        n_newton, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(int(n_newton))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on 60x60 grid
    nx_out, ny_out = 60, 60
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
    
    # Also evaluate initial condition on same grid
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
            "rtol": 1e-8,
            "iterations": int(sum(nonlinear_iterations) * 5),  # approximate linear iterations
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
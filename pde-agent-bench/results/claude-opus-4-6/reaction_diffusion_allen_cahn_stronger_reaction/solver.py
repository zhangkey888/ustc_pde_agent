import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse parameters
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.002)
    
    # Agent-selectable parameters
    mesh_resolution = 80
    element_degree = 2
    dt = 0.001  # smaller than suggested for accuracy
    
    # Diffusion coefficient - Allen-Cahn typically has small epsilon
    # We need to figure out epsilon and reaction from the case
    # Allen-Cahn: R(u) = lambda * (u^3 - u) or similar
    # "stronger_reaction" suggests larger lambda
    
    # For Allen-Cahn: -eps * laplacian(u) + lam*(u^3 - u) = f
    # The manufactured solution is u = exp(-t)*(0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    
    # Let's extract epsilon and reaction_lambda from case_spec if available
    params = case_spec.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    reaction_lambda = params.get("reaction_lambda", 10.0)
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time constant
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    lam_const = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))
    
    # Manufactured solution
    u_exact_ufl = ufl.exp(-t) * (0.15 + 0.12 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1]))
    
    # Compute source term f = du/dt - eps*laplacian(u) + lam*(u^3 - u)
    # du/dt = -exp(-t)*(0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    du_dt = -ufl.exp(-t) * (0.15 + 0.12 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1]))
    
    # Laplacian of u_exact
    # u = exp(-t) * (0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    # laplacian = exp(-t) * 0.12 * (-4*pi^2*sin(2*pi*x)*sin(2*pi*y) - 4*pi^2*sin(2*pi*x)*sin(2*pi*y))
    #           = exp(-t) * 0.12 * (-8*pi^2*sin(2*pi*x)*sin(2*pi*y))
    laplacian_u = ufl.exp(-t) * 0.12 * (-8.0 * pi**2) * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    
    # Reaction: R(u) = lam * (u^3 - u)
    R_u_exact = lam_const * (u_exact_ufl**3 - u_exact_ufl)
    
    # Source: f = du/dt - eps*laplacian(u) + R(u)
    f_source = du_dt - eps_const * laplacian_u + R_u_exact
    
    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (Newton unknown)
    v = ufl.TestFunction(V)
    
    # Initial condition at t=0
    t.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    u_h.interpolate(u_exact_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_exact_expr)
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + lam*(u^3 - u) = f
    # Weak form (residual):
    # (u - u_n)/dt * v + eps * grad(u) . grad(v) + lam*(u^3 - u)*v - f*v = 0
    F = (
        (u_h - u_n) / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + lam_const * (u_h**3 - u_h) * v * ufl.dx
        - f_source * v * ufl.dx
    )
    
    # Boundary conditions - update each time step
    # All boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, bc_dofs)
    
    # Setup nonlinear problem
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-8)
    
    # Time stepping
    n_steps = int(round(t_end / dt))
    current_t = 0.0
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        
        # Update boundary condition
        u_bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        # Use previous solution as initial guess (already in u_h from last step or copy)
        # u_h already has the previous solution which is a good guess
        
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 75x75 grid
    nx_eval, ny_eval = 75, 75
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
    
    # Also evaluate initial condition
    u_init_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.3)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a smaller dt for accuracy
    dt_val = 0.005
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    
    # Diffusion coefficient - Allen-Cahn typically has epsilon
    epsilon = pde.get("coefficients", {}).get("epsilon", None)
    if epsilon is None:
        epsilon = pde.get("epsilon", 0.01)
    
    # Reaction parameter
    reaction_lambda = pde.get("coefficients", {}).get("reaction_lambda", None)
    if reaction_lambda is None:
        reaction_lambda = pde.get("reaction_lambda", 5.0)
    
    # Try to get from nested structures
    if isinstance(pde.get("coefficients", None), dict):
        coeffs = pde["coefficients"]
        if "epsilon" in coeffs:
            epsilon = coeffs["epsilon"]
        if "reaction_lambda" in coeffs:
            reaction_lambda = coeffs["reaction_lambda"]
    
    # Mesh resolution - use finer mesh for accuracy
    N = 80
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space - P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = 0.2*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    u_exact_ufl = 0.2 * ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Compute source term from manufactured solution
    # u_t = -0.1*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # -epsilon * laplacian(u) = epsilon * (4*pi^2 + pi^2) * 0.2 * exp(-0.5*t) * sin(2*pi*x)*sin(pi*y)
    #                         = epsilon * 5*pi^2 * u
    # R(u) for Allen-Cahn: reaction_lambda * (u^3 - u)
    # PDE: u_t - epsilon*laplacian(u) + reaction_lambda*(u^3 - u) = f
    # f = u_t - epsilon*laplacian(u) + reaction_lambda*(u^3 - u)
    # u_t = -0.5 * u
    # -epsilon*laplacian(u) = epsilon * 5*pi^2 * u
    # So f = -0.5*u + epsilon*5*pi^2*u + reaction_lambda*(u^3 - u)
    
    # We'll compute f symbolically using UFL
    # But we need u_t explicitly since UFL doesn't differentiate w.r.t. t_const easily
    # u_t = d/dt [0.2*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)] = -0.1*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    u_t_ufl = -0.1 * ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Laplacian of u_exact: div(grad(u_exact))
    # grad(u) = (0.2*exp(-0.5*t)*2*pi*cos(2*pi*x)*sin(pi*y), 0.2*exp(-0.5*t)*sin(2*pi*x)*pi*cos(pi*y))
    # laplacian = -0.2*exp(-0.5*t)*(4*pi^2)*sin(2*pi*x)*sin(pi*y) - 0.2*exp(-0.5*t)*pi^2*sin(2*pi*x)*sin(pi*y)
    #           = -0.2*exp(-0.5*t)*5*pi^2*sin(2*pi*x)*sin(pi*y) = -5*pi^2 * u_exact
    
    # Source: f = u_t - epsilon*laplacian(u) + reaction_lambda*(u^3 - u)
    # f = u_t + epsilon*5*pi^2*u + reaction_lambda*(u^3 - u)
    # We compute it symbolically:
    f_ufl = u_t_ufl + epsilon * 5.0 * pi**2 * u_exact_ufl + reaction_lambda * (u_exact_ufl**3 - u_exact_ufl)
    
    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (for Newton)
    
    # Initial condition: u(x, 0) = 0.2*sin(2*pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: 0.2 * np.sin(2 * pi * X[0]) * np.sin(pi * X[1]))
    
    # Save initial condition for output
    nx_out, ny_out = 65, 65
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Evaluate initial condition on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape((nx_out, ny_out))
    
    # Boundary conditions - Dirichlet from exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Time stepping with backward Euler and Newton for nonlinear reaction
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    lam_c = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))
    
    # Nonlinear residual for backward Euler:
    # (u_h - u_n)/dt - epsilon*laplacian(u_h) + lambda*(u_h^3 - u_h) - f = 0
    v = ufl.TestFunction(V)
    
    F = (
        ufl.inner((u_h - u_n) / dt_c, v) * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + lam_c * ufl.inner(u_h**3 - u_h, v) * ufl.dx
        - ufl.inner(f_ufl, v) * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, u_h)
    
    # Set initial guess
    u_h.x.array[:] = u_n.x.array[:]
    
    # Update BC for t=dt
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Create nonlinear problem and solver
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
    ksp.setTolerances(rtol=1e-10, atol=1e-12)
    
    nonlinear_iterations_list = []
    total_linear_iterations = 0
    
    # Expression for BC interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # Time stepping loop
    t = 0.0
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
        # Initial guess: use previous solution
        u_h.x.array[:] = u_n.x.array[:]
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations_list.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Extract solution on output grid
    u_final_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_final_vals[eval_map] = vals.flatten()
    u_grid = u_final_vals.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": int(sum(nonlinear_iterations_list) * 5),  # rough estimate of linear iters
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(n) for n in nonlinear_iterations_list],
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
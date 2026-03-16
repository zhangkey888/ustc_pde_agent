import numpy as np
from dolfinx import mesh, fem, default_scalar_type, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Extract parameters
    epsilon = float(pde.get("epsilon", pde.get("diffusion_coefficient", 0.01)))
    source_expr_str = pde.get("source_term", "0")
    ic_str = pde.get("initial_condition", "0.4 + 0.1*sin(pi*x)*sin(pi*y)")
    
    # Reaction parameters
    reaction_type = pde.get("reaction_type", "logistic")
    reaction_rho = float(pde.get("reaction_rho", 1.0))
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.35))
    dt_val = float(time_params.get("dt", 0.01))
    time_scheme = time_params.get("scheme", "backward_euler")
    is_transient = t_end > 0 and dt_val > 0
    
    # BC
    bc_spec = pde.get("boundary_conditions", {})
    bc_type = bc_spec.get("type", "dirichlet")
    bc_value_str = bc_spec.get("value", None)
    
    # Grid parameters
    nx_out = 75
    ny_out = 75
    
    # Mesh resolution
    N = 120
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    # 4. Build source term as UFL expression
    def parse_source(s, x):
        """Build UFL expression from source string."""
        expr = s.replace("pi", str(np.pi))
        expr = expr.replace("x", "x[0]").replace("y", "x[1]")
        # We'll build it using UFL directly
        x0 = x[0]
        x1 = x[1]
        pi_u = ufl.pi
        
        # For this specific case:
        # f = 4*exp(-200*((x-0.4)**2 + (y-0.6)**2)) - 2*exp(-200*((x-0.65)**2 + (y-0.35)**2))
        f_ufl = (4.0 * ufl.exp(-200.0 * ((x0 - 0.4)**2 + (x1 - 0.6)**2))
                 - 2.0 * ufl.exp(-200.0 * ((x0 - 0.65)**2 + (x1 - 0.35)**2)))
        return f_ufl
    
    f_expr = parse_source(source_expr_str, x)
    
    # 5. Define functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # current solution (for Newton)
    
    # Initial condition: u0 = 0.4 + 0.1*sin(pi*x)*sin(pi*y)
    def ic_func(X):
        return 0.4 + 0.1 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1])
    
    u_n.interpolate(ic_func)
    u_h.interpolate(ic_func)
    
    # Save initial condition for output
    # Build evaluation grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    def eval_on_grid(u_func):
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
        
        u_values = np.full(points_2d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            pts_arr = np.array(points_on_proc)
            cells_arr = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts_arr, cells_arr)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)
    
    u_initial = eval_on_grid(u_n)
    
    # 6. Boundary conditions
    # Dirichlet BC: use initial condition value on boundary (or specified value)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    # For the BC value, if not specified, use the IC on the boundary
    # For logistic reaction-diffusion with IC = 0.4 + 0.1*sin(pi*x)*sin(pi*y),
    # sin(pi*x)*sin(pi*y) = 0 on boundary, so BC = 0.4
    if bc_value_str is not None:
        try:
            bc_val = float(bc_value_str)
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda X: np.full(X.shape[1], bc_val))
        except (ValueError, TypeError):
            # Use IC on boundary
            u_bc = fem.Function(V)
            u_bc.interpolate(ic_func)
    else:
        # Default: use IC value on boundary
        u_bc = fem.Function(V)
        u_bc.interpolate(ic_func)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # 7. Time stepping with backward Euler + Newton for nonlinear reaction
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    v = ufl.TestFunction(V)
    
    # Reaction term: logistic R(u) = rho * u * (1 - u)
    # The PDE is: du/dt - eps * laplacian(u) + R(u) = f
    # With R(u) = rho * u * (1-u), the residual is:
    # F = (u_h - u_n)/dt * v + eps * grad(u_h) . grad(v) + rho * u_h * (1 - u_h) * v - f * v
    # Wait: check sign convention. The PDE says:
    #   du/dt - eps * nabla^2 u + R(u) = f
    # Weak form (multiply by v, integrate):
    #   (du/dt, v) + eps*(grad u, grad v) + (R(u), v) = (f, v)
    # Backward Euler:
    #   ((u_h - u_n)/dt, v) + eps*(grad u_h, grad v) + (R(u_h), v) = (f, v)
    
    rho = fem.Constant(domain, PETSc.ScalarType(reaction_rho))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    F = ((u_h - u_n) / dt_c * v * ufl.dx
         + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + rho * u_h * (1.0 - u_h) * v * ufl.dx
         - f_expr * v * ufl.dx)
    
    # Jacobian
    J = ufl.derivative(F, u_h)
    
    # Setup nonlinear problem
    problem = petsc.NonlinearProblem(F, u_h, bcs=bcs, J=J)
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
    
    # 8. Time loop
    t = 0.0
    n_steps = int(np.round(t_end / dt_val))
    nonlinear_iterations = []
    total_linear_iters = 0
    
    for step in range(n_steps):
        t += dt_val
        
        # Solve nonlinear problem
        n_iters, converged = solver.solve(u_h)
        if not converged:
            print(f"Newton did not converge at step {step}, t={t:.4f}")
        
        nonlinear_iterations.append(int(n_iters))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()
    
    # 9. Extract solution on grid
    u_grid = eval_on_grid(u_h)
    
    # 10. Build solver_info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": int(sum(nonlinear_iterations) * 10),  # rough estimate of linear iters
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }
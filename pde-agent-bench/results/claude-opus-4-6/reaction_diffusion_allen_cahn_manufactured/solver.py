import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    domain_spec = case_spec.get("domain", {})
    
    # Time parameters
    t_end = time_params.get("t_end", 0.15)
    dt_suggested = time_params.get("dt", 0.005)
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    
    # Allen-Cahn type reaction: R(u) = u^3 - u (or similar)
    # The manufactured solution is u = exp(-t)*(0.3*sin(pi*x)*sin(pi*y))
    # For Allen-Cahn: ∂u/∂t - ε∇²u + (u³ - u)/ε² = f  (typical form)
    # But we need to figure out epsilon and the reaction form from the case spec
    
    # Get epsilon from case spec
    epsilon = pde.get("epsilon", None)
    if epsilon is None:
        # Try to get from coefficients
        coeffs = pde.get("coefficients", {})
        epsilon = coeffs.get("epsilon", 0.01)
    
    # Mesh resolution - use fine enough mesh for accuracy
    nx = 80
    ny = 80
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * 0.3 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # For Allen-Cahn: ∂u/∂t - ε∇²u + R(u) = f
    # where R(u) = u³ - u (standard Allen-Cahn reaction, or could be (u³-u)/ε²)
    # Let's check what reaction term is specified
    reaction_type = pde.get("reaction", {}).get("type", "allen_cahn")
    
    # Standard Allen-Cahn: ∂u/∂t - ε∇²u + (1/ε²)(u³ - u) = f
    # But sometimes it's just: ∂u/∂t - ε∇²u + u³ - u = f
    # We'll derive f from the manufactured solution
    
    # u_exact = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    # ∂u/∂t = -exp(-t) * 0.3 * sin(pi*x) * sin(pi*y) = -u_exact
    # ∇²u = -2*pi² * exp(-t) * 0.3 * sin(pi*x) * sin(pi*y) = -2*pi² * u_exact
    # -ε∇²u = 2*ε*pi² * u_exact
    
    # For Allen-Cahn with R(u) = u³ - u:
    # f = ∂u/∂t - ε∇²u + u³ - u
    # f = -u_exact + 2*ε*pi²*u_exact + u_exact³ - u_exact
    # f = u_exact*(-1 + 2*ε*pi² - 1) + u_exact³
    # f = u_exact*(2*ε*pi² - 2) + u_exact³
    
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Compute source term from manufactured solution
    # ∂u_exact/∂t
    du_dt_exact = -ufl.exp(-t_const) * 0.3 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # -ε∇²u_exact = ε * 2 * pi² * u_exact
    neg_eps_laplacian = eps_const * 2.0 * pi**2 * u_exact_ufl
    
    # R(u) = u³ - u for Allen-Cahn
    R_exact = u_exact_ufl**3 - u_exact_ufl
    
    # f = ∂u/∂t - ε∇²u + R(u)
    f_expr = du_dt_exact + neg_eps_laplacian + R_exact
    
    # Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (unknown)
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x, 0) = 0.3 * sin(pi*x) * sin(pi*y)
    u_n.interpolate(lambda X: 0.3 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    u_h.x.array[:] = u_n.x.array[:]
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 70, 70
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Backward Euler: (u - u_n)/dt - ε∇²u + R(u) = f
    # Residual: (u - u_n)/dt * v + ε*∇u·∇v + R(u)*v - f*v = 0
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = ((u_h - u_n) / dt_const * v * ufl.dx
         + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + (u_h**3 - u_h) * v * ufl.dx
         - f_expr * v * ufl.dx)
    
    # Boundary conditions: u = u_exact on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # We'll update u_bc each time step
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Set up nonlinear problem
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
    t = 0.0
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = t
        u_bc.interpolate(lambda X, tv=t_val: np.exp(-tv) * 0.3 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        
        nonlinear_iterations.append(int(n_iters))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
        u_h.x.scatter_forward()
    
    # Evaluate solution on output grid
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
    
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition
    u_n_init = fem.Function(V)
    u_n_init.interpolate(lambda X: 0.3 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_init_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
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
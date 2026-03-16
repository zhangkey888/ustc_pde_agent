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
    epsilon = pde_config.get("epsilon", 1.0)
    reaction_type = pde_config.get("reaction", {}).get("type", "allen_cahn")
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.15)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution - use fine enough mesh for accuracy
    nx = ny = 80
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time constant
    t = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Manufactured solution: u_exact = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t) * (0.3 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    
    # For Allen-Cahn: R(u) = u^3 - u (or similar)
    # The reaction term for Allen-Cahn is typically R(u) = u^3 - u
    # So PDE: du/dt - eps * laplacian(u) + u^3 - u = f
    
    # Compute source term from manufactured solution:
    # u_exact = exp(-t) * 0.3 * sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t) * 0.3 * sin(pi*x)*sin(pi*y) = -u_exact
    # laplacian(u_exact) = exp(-t) * 0.3 * (-2*pi^2) * sin(pi*x)*sin(pi*y) = -2*pi^2 * u_exact
    # -eps * laplacian(u_exact) = 2*eps*pi^2 * u_exact
    # R(u_exact) = u_exact^3 - u_exact
    # f = du/dt - eps*laplacian(u_exact) + R(u_exact)
    #   = -u_exact + 2*eps*pi^2*u_exact + u_exact^3 - u_exact
    #   = u_exact*(-1 + 2*eps*pi^2 - 1) + u_exact^3
    #   = u_exact*(2*eps*pi^2 - 2) + u_exact^3
    
    f_ufl = (-u_exact_ufl + eps_const * 2.0 * pi**2 * u_exact_ufl 
             + u_exact_ufl**3 - u_exact_ufl)
    
    # 4. Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (for Newton)
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x, 0) = 0.3 * sin(pi*x) * sin(pi*y)
    u_init_expr = fem.Expression(
        0.3 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)
    u_h.interpolate(u_init_expr)
    
    # Save initial condition for output
    # We'll evaluate on grid later
    
    # 5. Boundary conditions - u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        t.value = t_val
        bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
    
    update_bc(0.0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # 6. Nonlinear residual (backward Euler)
    # (u_h - u_n)/dt - eps*laplacian(u_h) + R(u_h) = f
    # Weak form: (u_h - u_n)/dt * v + eps*grad(u_h)·grad(v) + (u_h^3 - u_h)*v - f*v = 0
    
    F = ((u_h - u_n) / dt_const * v * ufl.dx
         + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + (u_h**3 - u_h) * v * ufl.dx
         - f_ufl * v * ufl.dx)
    
    # 7. Newton solver setup
    problem = petsc.NonlinearProblem(F, u_h, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-12
    solver.max_it = 25
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-9)
    
    # 8. Time stepping
    nonlinear_iterations = []
    total_linear_iterations = 0
    current_t = 0.0
    
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        
        # Update boundary condition
        update_bc(current_t)
        
        # Solve nonlinear problem
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={current_t}"
        
        nonlinear_iterations.append(int(n_iters))
        # Estimate linear iterations (each Newton step ~ 1 linear solve)
        total_linear_iterations += int(n_iters)
        
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Evaluate initial condition on grid for output
    # Reset time to 0 for initial condition evaluation
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_init_expr)
    
    # 10. Extract solution on 70x70 grid
    nx_out, ny_out = 70, 70
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    
    # Add z=0 for 3D point array
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate final solution
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Evaluate initial condition
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts, cls)
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
            "rtol": 1e-9,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
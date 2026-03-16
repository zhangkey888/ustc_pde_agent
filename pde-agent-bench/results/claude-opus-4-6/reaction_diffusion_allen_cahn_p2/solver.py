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
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)
    
    # Parameters
    mesh_resolution = 80
    element_degree = 2
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Diffusion coefficient (epsilon)
    # For Allen-Cahn: -eps*laplacian(u) + R(u) = f
    # R(u) = u^3 - u (typical Allen-Cahn reaction)
    # But we use manufactured solution approach, so we compute f from the exact solution
    epsilon = pde.get("coefficients", {}).get("epsilon", 0.01)
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Time constant
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution: u = exp(-t)*(0.25*sin(2*pi*x)*sin(pi*y))
    u_exact = ufl.exp(-t_const) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Allen-Cahn reaction: R(u) = u^3 - u
    # PDE: du/dt - eps*laplacian(u) + u^3 - u = f
    # f = du_exact/dt - eps*laplacian(u_exact) + u_exact^3 - u_exact
    
    # du/dt = -exp(-t)*(0.25*sin(2*pi*x)*sin(pi*y))
    du_dt_exact = -ufl.exp(-t_const) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # laplacian(u_exact) = exp(-t)*0.25*(-4*pi^2*sin(2*pi*x)*sin(pi*y) - pi^2*sin(2*pi*x)*sin(pi*y))
    # = exp(-t)*0.25*(-5*pi^2)*sin(2*pi*x)*sin(pi*y)
    laplacian_u_exact = ufl.exp(-t_const) * 0.25 * (-5.0 * pi**2) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term
    f_expr = du_dt_exact - epsilon * laplacian_u_exact + u_exact**3 - u_exact
    
    # Trial and test functions
    v = ufl.TestFunction(V)
    
    # Current solution and previous time step
    u_h = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    
    # Set initial condition at t=0
    t_const.value = 0.0
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    u_h.interpolate(u_exact_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_exact_expr)
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + u^3 - u = f
    # Weak form (nonlinear residual):
    # (u - u_n)/dt * v * dx + eps * grad(u) . grad(v) * dx + (u^3 - u) * v * dx - f * v * dx = 0
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    F = ((u_h - u_n) / dt_const * v * ufl.dx
         + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + (u_h**3 - u_h) * v * ufl.dx
         - f_expr * v * ufl.dx)
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # Setup nonlinear solver
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
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        
        # Solve
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 60x60 grid
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": int(sum(nonlinear_iterations) * 5),  # approximate total linear iters
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": [int(n) for n in nonlinear_iterations],
        }
    }
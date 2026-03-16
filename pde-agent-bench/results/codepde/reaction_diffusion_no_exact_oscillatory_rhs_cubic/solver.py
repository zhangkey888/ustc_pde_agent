import numpy as np
from dolfinx import mesh, fem, default_scalar_type, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde_config.get("epsilon", 0.01)
    reaction_type = pde_config.get("reaction_type", "cubic")
    reaction_params = pde_config.get("reaction_params", {})
    source_expr_str = pde_config.get("source_term", "sin(6*pi*x)*sin(5*pi*y)")
    bc_type = pde_config.get("bc_type", "dirichlet")
    bc_value = pde_config.get("bc_value", 0.0)
    
    # Time parameters
    time_params = pde_config.get("time", None)
    is_transient = time_params is not None
    
    if is_transient:
        t_end = time_params.get("t_end", 0.3)
        dt_suggested = time_params.get("dt", 0.005)
        scheme = time_params.get("scheme", "backward_euler")
    else:
        t_end = 0.0
        dt_suggested = 0.005
        scheme = "backward_euler"
    
    # Initial condition
    ic_str = pde_config.get("initial_condition", "0.2*sin(3*pi*x)*sin(2*pi*y)")
    
    # 2. Create mesh
    nx_mesh = 100
    ny_mesh = 100
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # 4. Source term
    f_expr = ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
    
    # 5. Boundary conditions - homogeneous Dirichlet
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_all(xx):
        return (np.isclose(xx[0], 0.0) | np.isclose(xx[0], 1.0) |
                np.isclose(xx[1], 0.0) | np.isclose(xx[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    if isinstance(bc_value, (int, float)):
        bc = fem.dirichletbc(PETSc.ScalarType(bc_value), dofs, V)
    else:
        bc_func = fem.Function(V)
        bc_func.interpolate(lambda xx: np.full(xx.shape[1], float(bc_value)))
        bc = fem.dirichletbc(bc_func, dofs)
    
    bcs = [bc]
    
    # 6. Define reaction term R(u)
    # For cubic reaction: R(u) = u^3 (common choice)
    # Could also be u*(1-u), etc.
    def reaction(u_val):
        coeff = reaction_params.get("coefficient", 1.0)
        if reaction_type == "cubic":
            return coeff * u_val**3
        elif reaction_type == "linear":
            return coeff * u_val
        elif reaction_type == "fisher":
            return coeff * u_val * (1.0 - u_val)
        else:
            return coeff * u_val**3
    
    def reaction_derivative(u_val):
        coeff = reaction_params.get("coefficient", 1.0)
        if reaction_type == "cubic":
            return 3.0 * coeff * u_val**2
        elif reaction_type == "linear":
            return coeff
        elif reaction_type == "fisher":
            return coeff * (1.0 - 2.0 * u_val)
        else:
            return 3.0 * coeff * u_val**2
    
    # Use dt
    dt = dt_suggested
    
    if is_transient:
        # Time-dependent solve with backward Euler and Newton for nonlinear reaction
        u_n = fem.Function(V, name="u_n")  # solution at previous time step
        u_h = fem.Function(V, name="u_h")  # current solution (Newton iterate)
        v = ufl.TestFunction(V)
        
        # Initial condition
        u_n.interpolate(lambda xx: 0.2 * np.sin(3 * pi * xx[0]) * np.sin(2 * pi * xx[1]))
        u_h.x.array[:] = u_n.x.array[:]
        
        # Store initial condition for output
        # We'll evaluate it at the end
        
        # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + R(u) = f
        # Residual F = (u - u_n)/dt * v + eps * grad(u) . grad(v) + R(u)*v - f*v = 0
        dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
        eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
        
        F = ((u_h - u_n) / dt_const * v * ufl.dx
             + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
             + reaction(u_h) * v * ufl.dx
             - f_expr * v * ufl.dx)
        
        # Setup nonlinear problem
        problem = petsc.NonlinearProblem(F, u_h, bcs=bcs)
        solver = nls.petsc.NewtonSolver(domain.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-8
        solver.atol = 1e-10
        solver.max_it = 25
        solver.relaxation_parameter = 1.0
        
        ksp = solver.krylov_solver
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        
        # Time stepping
        t = 0.0
        n_steps = int(np.round(t_end / dt))
        dt_actual = t_end / n_steps  # adjust dt to hit t_end exactly
        dt_const.value = dt_actual
        
        nonlinear_iterations = []
        total_linear_iters = 0
        
        for step in range(n_steps):
            t += dt_actual
            
            # Use previous solution as initial guess
            u_h.x.array[:] = u_n.x.array[:]
            
            n_iters, converged = solver.solve(u_h)
            assert converged, f"Newton solver did not converge at step {step}, t={t:.4f}"
            u_h.x.scatter_forward()
            
            nonlinear_iterations.append(n_iters)
            # Approximate linear iterations (each Newton step ~ 1 linear solve)
            total_linear_iters += n_iters
            
            # Update previous solution
            u_n.x.array[:] = u_h.x.array[:]
        
        u_solution = u_h
        
    else:
        # Steady-state solve
        u_h = fem.Function(V, name="u_h")
        v = ufl.TestFunction(V)
        eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
        
        F = (eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
             + reaction(u_h) * v * ufl.dx
             - f_expr * v * ufl.dx)
        
        problem = petsc.NonlinearProblem(F, u_h, bcs=bcs)
        solver = nls.petsc.NewtonSolver(domain.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-8
        solver.atol = 1e-10
        solver.max_it = 30
        
        ksp = solver.krylov_solver
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        
        n_iters, converged = solver.solve(u_h)
        assert converged
        u_h.x.scatter_forward()
        
        nonlinear_iterations = [n_iters]
        total_linear_iters = n_iters
        n_steps = 0
        dt_actual = 0.0
        
        u_solution = u_h
    
    # 7. Extract solution on uniform grid
    nx_out = 70
    ny_out = 70
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
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
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_solution.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition if transient
    result = {"u": u_grid}
    
    if is_transient:
        # Evaluate initial condition on grid
        u_ic = fem.Function(V)
        u_ic.interpolate(lambda xx: 0.2 * np.sin(3 * pi * xx[0]) * np.sin(2 * pi * xx[1]))
        
        u_ic_values = np.full(points_3d.shape[0], np.nan)
        if len(points_on_proc) > 0:
            vals_ic = u_ic.eval(pts_arr, cells_arr)
            u_ic_values[eval_map] = vals_ic.flatten()
        
        u_initial_grid = u_ic_values.reshape((nx_out, ny_out))
        result["u_initial"] = u_initial_grid
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_linear_iters,
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    if is_transient:
        solver_info["dt"] = dt_actual if is_transient else dt_suggested
        solver_info["n_steps"] = n_steps
        solver_info["time_scheme"] = scheme
    
    result["solver_info"] = solver_info
    
    return result
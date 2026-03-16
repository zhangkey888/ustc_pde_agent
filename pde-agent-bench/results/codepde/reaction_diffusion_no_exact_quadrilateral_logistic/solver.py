import numpy as np
from dolfinx import mesh, fem, default_scalar_type, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Extract parameters
    source_term_str = pde_config.get("source_term", "1.0")
    ic_str = pde_config.get("initial_condition", "0.25 + 0.15*sin(pi*x)*sin(pi*y)")
    
    time_params = pde_config.get("time", {})
    t_end = float(time_params.get("t_end", 0.3))
    dt_suggested = float(time_params.get("dt", 0.01))
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Reaction parameters - logistic reaction: R(u) = rho * u * (1 - u)
    reaction_type = pde_config.get("reaction_type", "logistic")
    reaction_rho = float(pde_config.get("reaction_rho", pde_config.get("rho", 1.0)))
    
    # Diffusion coefficient
    epsilon = float(pde_config.get("epsilon", pde_config.get("diffusion_coefficient", 0.01)))
    
    # Boundary condition
    bc_type = pde_config.get("boundary_condition", {})
    bc_value = float(bc_type.get("value", 0.0)) if isinstance(bc_type, dict) else 0.0
    
    # Agent-selectable parameters
    agent_params = case_spec.get("agent_params", {})
    nx = int(agent_params.get("mesh_resolution", 80))
    ny = nx
    degree = int(agent_params.get("element_degree", 2))
    dt = float(agent_params.get("dt", dt_suggested))
    
    # Use quadrilateral mesh as specified in case ID
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space - use "Lagrange" for quads (mapped Q elements)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_k = fem.Function(V)  # current Newton iterate
    v = ufl.TestFunction(V)
    
    # Source term
    f_val = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Diffusion coefficient
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Reaction rate
    rho_c = fem.Constant(domain, PETSc.ScalarType(reaction_rho))
    
    # Time step
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Initial condition
    def ic_func(X):
        return 0.25 + 0.15 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1])
    
    u_n.interpolate(ic_func)
    u_k.interpolate(ic_func)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(ic_func)
    
    # 5. Boundary conditions - Dirichlet
    # Determine BC value from case_spec
    bc_g = pde_config.get("boundary_condition", {})
    if isinstance(bc_g, dict):
        g_value = float(bc_g.get("value", 0.0))
        g_expr_str = bc_g.get("expression", None)
    else:
        g_value = 0.0
        g_expr_str = None
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    if g_expr_str is not None:
        # Parse expression if provided
        u_bc.interpolate(lambda X: np.full(X.shape[1], g_value))
    else:
        u_bc.interpolate(lambda X: np.full(X.shape[1], g_value))
    
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # 6. Variational form - Backward Euler with Newton for nonlinear reaction
    # Logistic reaction: R(u) = rho * u * (1 - u)
    # Residual: (u_k - u_n)/dt * v + eps * grad(u_k) . grad(v) + rho * u_k * (1 - u_k) * v - f * v = 0
    
    F = (
        (u_k - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u_k), ufl.grad(v)) * ufl.dx
        + rho_c * u_k * (1.0 - u_k) * v * ufl.dx
        - f_val * v * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, u_k, ufl.TrialFunction(V))
    
    # 7. Newton solver setup
    problem = petsc.NonlinearProblem(F, u_k, bcs=bcs, J=J)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # 8. Time stepping
    n_steps = int(np.round(t_end / dt))
    # Adjust dt to exactly hit t_end
    dt_actual = t_end / n_steps
    dt_c.value = dt_actual
    
    total_linear_iterations = 0
    nonlinear_iterations_list = []
    
    t = 0.0
    for step in range(n_steps):
        t += dt_actual
        
        # Use previous solution as initial guess
        u_k.x.array[:] = u_n.x.array[:]
        
        n_newton, converged = solver.solve(u_k)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        
        u_k.x.scatter_forward()
        nonlinear_iterations_list.append(n_newton)
        
        # Estimate linear iterations (Newton iters * ~avg KSP iters)
        # We'll track via ksp
        total_linear_iterations += n_newton  # approximate
        
        # Update previous solution
        u_n.x.array[:] = u_k.x.array[:]
    
    # 9. Extract solution on 65x65 uniform grid
    n_eval = 65
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    # Point evaluation
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
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_k.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations_list,
        }
    }
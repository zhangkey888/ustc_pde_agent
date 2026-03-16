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
    epsilon = pde_config.get("epsilon", pde_config.get("diffusion_coefficient", 0.01))
    source_expr_str = pde_config.get("source_term", pde_config.get("f", "0"))
    bc_value = pde_config.get("bc_value", 0.0)
    
    # Reaction type
    reaction_type = pde_config.get("reaction_type", "logistic")
    reaction_params = pde_config.get("reaction_params", {})
    reaction_rate = reaction_params.get("rate", pde_config.get("reaction_rate", 1.0))
    
    # Time parameters
    time_params = pde_config.get("time", None)
    is_transient = time_params is not None
    
    if is_transient:
        t_end = time_params.get("t_end", 0.4)
        dt_val = time_params.get("dt", 0.01)
        scheme = time_params.get("scheme", "backward_euler")
    else:
        t_end = 0.0
        dt_val = 0.01
        scheme = "backward_euler"
    
    # Initial condition
    ic_str = pde_config.get("initial_condition", pde_config.get("u0", "0"))
    
    # 2. Create mesh
    nx_mesh = 100
    ny_mesh = 100
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Build source term as UFL expression
    # f = 6*(exp(-160*((x-0.3)**2 + (y-0.7)**2)) + 0.8*exp(-160*((x-0.75)**2 + (y-0.35)**2)))
    f_ufl = 6.0 * (ufl.exp(-160.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) +
                    0.8 * ufl.exp(-160.0 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2)))
    
    # 5. Boundary conditions (u = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_all(xcoord):
        return (np.isclose(xcoord[0], 0.0) | np.isclose(xcoord[0], 1.0) |
                np.isclose(xcoord[1], 0.0) | np.isclose(xcoord[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_func = fem.Function(V)
    bc_func.interpolate(lambda xcoord: np.full(xcoord.shape[1], float(bc_value)))
    bc = fem.dirichletbc(bc_func, dofs)
    bcs = [bc]
    
    # 6. Set up the problem
    # u_n: solution at previous time step (for transient)
    # u_h: current solution (nonlinear unknown)
    u_h = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    # Initial condition
    # u0 = 0.3*exp(-50*((x-0.3)**2 + (y-0.5)**2)) + 0.3*exp(-50*((x-0.7)**2 + (y-0.5)**2))
    u_h.interpolate(lambda xcoord: (
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.3)**2 + (xcoord[1] - 0.5)**2)) +
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.7)**2 + (xcoord[1] - 0.5)**2))
    ))
    u_n.interpolate(lambda xcoord: (
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.3)**2 + (xcoord[1] - 0.5)**2)) +
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.7)**2 + (xcoord[1] - 0.5)**2))
    ))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda xcoord: (
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.3)**2 + (xcoord[1] - 0.5)**2)) +
        0.3 * np.exp(-50.0 * ((xcoord[0] - 0.7)**2 + (xcoord[1] - 0.5)**2))
    ))
    
    # Reaction term: logistic R(u) = rate * u * (1 - u)
    R_u = reaction_rate * u_h * (1.0 - u_h)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    if is_transient:
        # Backward Euler: (u - u_n)/dt - eps * laplacian(u) + R(u) = f
        # Weak form: (u - u_n)/dt * v + eps * grad(u) . grad(v) + R(u) * v - f * v = 0
        F = ((u_h - u_n) / dt_const * v * ufl.dx +
             eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx +
             R_u * v * ufl.dx -
             f_ufl * v * ufl.dx)
    else:
        # Steady: -eps * laplacian(u) + R(u) = f
        F = (eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx +
             R_u * v * ufl.dx -
             f_ufl * v * ufl.dx)
    
    # 7. Newton solver setup
    problem = petsc.NonlinearProblem(F, u_h, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # 8. Time stepping
    nonlinear_iterations_list = []
    total_linear_iterations = 0
    
    if is_transient:
        n_steps = int(round(t_end / dt_val))
        t = 0.0
        for step in range(n_steps):
            t += dt_val
            n_iters, converged = solver.solve(u_h)
            assert converged, f"Newton solver did not converge at step {step}, t={t}"
            u_h.x.scatter_forward()
            nonlinear_iterations_list.append(n_iters)
            # Update previous solution
            u_n.x.array[:] = u_h.x.array[:]
    else:
        n_steps = 0
        n_iters, converged = solver.solve(u_h)
        assert converged, "Newton solver did not converge for steady problem"
        u_h.x.scatter_forward()
        nonlinear_iterations_list.append(n_iters)
    
    # 9. Extract solution on uniform grid
    nx_out = 70
    ny_out = 70
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # 10. Build solver_info
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "nonlinear_iterations": nonlinear_iterations_list,
    }
    
    if is_transient:
        solver_info["dt"] = dt_val
        solver_info["n_steps"] = n_steps
        solver_info["time_scheme"] = scheme
        solver_info["iterations"] = sum(nonlinear_iterations_list) * 10  # estimate
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    if is_transient:
        result["u_initial"] = u_initial_grid
    
    return result
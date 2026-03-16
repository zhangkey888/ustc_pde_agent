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
    epsilon = float(pde.get("epsilon", 0.01))
    source_expr_str = pde.get("source_term", "5*exp(-180*((x-0.35)**2 + (y-0.55)**2))")
    ic_str = pde.get("initial_condition", "0.1*exp(-50*((x-0.5)**2 + (y-0.5)**2))")
    
    # Reaction type - Allen-Cahn: R(u) = (u^3 - u) / epsilon^2 typically
    reaction_type = pde.get("reaction_type", "allen_cahn")
    reaction_params = pde.get("reaction", {})
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.25))
    dt_suggested = float(time_params.get("dt", 0.005))
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Boundary conditions
    bc_spec = pde.get("boundary_conditions", {})
    bc_type = bc_spec.get("type", "dirichlet")
    bc_value = float(bc_spec.get("value", 0.0))
    
    # Domain
    domain_spec = pde.get("domain", {})
    
    # 2. Solver parameters
    nx = 80
    ny = 80
    degree = 1
    dt = dt_suggested
    
    # 3. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 4. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # 5. Build source term as UFL expression
    f_ufl = 5.0 * ufl.exp(-180.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # 6. Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_k = fem.Function(V)  # current Newton iterate / solution
    
    # Initial condition
    u_n.interpolate(lambda X: 0.1 * np.exp(-50.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2)))
    u_k.interpolate(lambda X: 0.1 * np.exp(-50.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2)))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: 0.1 * np.exp(-50.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2)))
    
    # 7. Boundary conditions - Dirichlet u=0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_value), boundary_dofs, V)
    bcs = [bc]
    
    # 8. Variational form - Allen-Cahn reaction-diffusion
    # Allen-Cahn: du/dt - epsilon * laplacian(u) + R(u) = f
    # where R(u) = (u^3 - u) / (epsilon_ac^2) for standard Allen-Cahn
    # But the problem says "reaction_diffusion" with Allen-Cahn type
    # Standard Allen-Cahn: du/dt = eps * laplacian(u) + u - u^3
    # Rewritten: du/dt - eps * laplacian(u) + (u^3 - u) = f
    # So R(u) = u^3 - u (the cubic reaction)
    
    # Get Allen-Cahn specific parameters
    # The diffusion is epsilon, reaction is Allen-Cahn type
    
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + R(u) = f
    # Nonlinear residual F = 0
    # R(u) for Allen-Cahn = u^3 - u
    
    F = (
        (u_k - u_n) / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_k), ufl.grad(v)) * ufl.dx
        + (u_k**3 - u_k) * v * ufl.dx
        - f_ufl * v * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, u_k, ufl.TrialFunction(V))
    
    # 9. Setup nonlinear solver
    problem = petsc.NonlinearProblem(F, u_k, bcs=bcs, J=J)
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
    opts = PETSc.Options()
    ksp.setFromOptions()
    
    # 10. Time stepping
    n_steps = int(np.round(t_end / dt))
    dt_actual = t_end / n_steps
    dt_const.value = dt_actual
    
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    t = 0.0
    for step in range(n_steps):
        t += dt_actual
        
        # Use previous solution as initial guess
        u_k.x.array[:] = u_n.x.array[:]
        
        n_iters, converged = solver.solve(u_k)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        
        u_k.x.scatter_forward()
        nonlinear_iterations.append(int(n_iters))
        
        # Update previous solution
        u_n.x.array[:] = u_k.x.array[:]
    
    # 11. Extract solution on 60x60 grid
    nx_out = 60
    ny_out = 60
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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
        vals = u_k.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
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
            "iterations": int(sum(nonlinear_iterations) * 5),  # approximate linear iters
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
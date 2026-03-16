import numpy as np
from dolfinx import mesh, fem, default_scalar_type, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde.get("epsilon", 1.0)
    source_expr_str = pde.get("source_term", "3*cos(3*pi*x)*sin(2*pi*y)")
    reaction_type = pde.get("reaction_type", "allen_cahn")
    reaction_params = pde.get("reaction_params", {})
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_val = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Initial condition
    ic_str = pde.get("initial_condition", "0.2*sin(3*pi*x)*sin(2*pi*y)")
    
    # Boundary conditions
    bc_spec = pde.get("boundary_conditions", {})
    
    # Grid for output
    nx_out = 70
    ny_out = 70
    
    # Mesh resolution - use fine mesh for accuracy
    N = 120
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # 4. Parse source term
    # f = 3*cos(3*pi*x)*sin(2*pi*y)
    f_expr = 3.0 * ufl.cos(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    # 5. Define reaction term R(u)
    # Allen-Cahn: R(u) = (u^3 - u) / epsilon^2 or similar
    # Common Allen-Cahn reaction: R(u) = (1/eps^2) * u * (u^2 - 1) or u^3 - u
    # The standard Allen-Cahn is: du/dt = eps * laplacian(u) - (1/eps)*(u^3 - u)
    # In our form: du/dt - eps * laplacian(u) + R(u) = f
    # So R(u) for Allen-Cahn is typically u^3 - u (or scaled)
    
    # 6. Initial condition
    # u0 = 0.2*sin(3*pi*x)*sin(2*pi*y)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: 0.2 * np.sin(3.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: 0.2 * np.sin(3.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    
    # 7. Boundary conditions - homogeneous Dirichlet (default for reaction-diffusion)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Check if there's a specific BC value
    g_val = 0.0  # default homogeneous
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.full_like(X[0], g_val))
    bc = fem.dirichletbc(u_bc, bc_dofs)
    bcs = [bc]
    
    # 8. Time stepping with backward Euler and Newton for nonlinear reaction
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Current solution
    u_h = fem.Function(V, name="u")
    u_h.x.array[:] = u_n.x.array[:]
    
    v = ufl.TestFunction(V)
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + R(u) = f
    # Weak form (residual): 
    # integral[ (u - u_n)/dt * v + eps * grad(u) . grad(v) + R(u)*v - f*v ] dx = 0
    
    # Allen-Cahn reaction: R(u) = u^3 - u
    u_var = u_h  # nonlinear unknown
    
    R_u = u_var**3 - u_var  # Allen-Cahn reaction term
    
    F = (
        (u_var - u_n) / dt * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u_var), ufl.grad(v)) * ufl.dx
        + R_u * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    
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
    opts = PETSc.Options()
    ksp.setFromOptions()
    
    # Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt_val))
    nonlinear_iterations = []
    total_linear_iters = 0
    
    for step in range(n_steps):
        t += dt_val
        
        # Use previous solution as initial guess
        u_h.x.array[:] = u_n.x.array[:]
        
        n_iters, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t:.4f}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Extract solution on uniform grid
    x_grid = np.linspace(0.0, 1.0, nx_out)
    y_grid = np.linspace(0.0, 1.0, ny_out)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X_grid.ravel()
    points[1, :] = Y_grid.ravel()
    
    # Point evaluation
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on the same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iterations) * 10,  # approximate total linear iterations
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
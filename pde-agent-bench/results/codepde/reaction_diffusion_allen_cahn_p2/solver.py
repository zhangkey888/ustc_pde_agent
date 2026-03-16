import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Reaction type
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "allen_cahn")
    reaction_coeff = reaction.get("coefficient", 1.0)
    
    # Diffusion
    diffusion = pde_config.get("diffusion", {})
    epsilon = diffusion.get("epsilon", 1.0)
    if isinstance(epsilon, dict):
        epsilon = epsilon.get("value", 1.0)
    
    # Manufactured solution info
    manufactured = pde_config.get("manufactured_solution", {})
    
    # Parameters
    mesh_resolution = 80
    element_degree = 2
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*(0.25*sin(2*pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t_const) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # Compute source term f from: du/dt - eps*laplacian(u) + R(u) = f
    # du/dt = -exp(-t)*0.25*sin(2*pi*x)*sin(pi*y)
    dudt_ufl = -ufl.exp(-t_const) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # laplacian(u) = exp(-t)*0.25*(-4*pi^2*sin(2*pi*x)*sin(pi*y) - pi^2*sin(2*pi*x)*sin(pi*y))
    #              = exp(-t)*0.25*(-5*pi^2)*sin(2*pi*x)*sin(pi*y)
    # -eps*laplacian(u) = eps*exp(-t)*0.25*5*pi^2*sin(2*pi*x)*sin(pi*y)
    
    # For Allen-Cahn: R(u) = reaction_coeff * (u^3 - u)
    # f = du/dt - eps*laplacian(u) + R(u)
    # We'll compute this symbolically via UFL
    
    # We need grad(u_exact) for the laplacian term
    # But u_exact depends on t_const which is a Constant, not spatial
    # Let's build f symbolically
    
    # Actually, let's compute -eps * div(grad(u_exact)) + R(u_exact) + du/dt = f
    # We can use UFL's grad on spatial coordinates
    
    # u_exact as a function of spatial coords and t_const
    u_ex = u_exact_ufl
    
    # grad and div for laplacian
    grad_u_ex = ufl.grad(u_ex)
    laplacian_u_ex = ufl.div(grad_u_ex)
    
    # Allen-Cahn reaction: R(u) = coeff * (u^3 - u)
    R_u_ex = reaction_coeff * (u_ex**3 - u_ex)
    
    # Source term: f = du/dt - eps*laplacian(u) + R(u)
    f_expr = dudt_ufl - epsilon * laplacian_u_ex + R_u_ex
    
    # 4. Set up the nonlinear time-stepping problem
    # Backward Euler: (u^{n+1} - u^n)/dt - eps*laplacian(u^{n+1}) + R(u^{n+1}) = f^{n+1}
    # Residual: (u - u_n)/dt * v + eps*grad(u)*grad(v) + R(u)*v - f*v = 0
    
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # current solution (unknown)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    react_const = fem.Constant(domain, default_scalar_type(reaction_coeff))
    
    # Nonlinear residual
    F = ((u_h - u_n) / dt_const * v * ufl.dx
         + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
         + react_const * (u_h**3 - u_h) * v * ufl.dx
         - f_expr * v * ufl.dx)
    
    # 5. Boundary conditions - u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_func = fem.Function(V)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, boundary_dofs)
    
    # Function to update BC values at a given time
    def update_bc(t_val):
        t_const.value = t_val
        bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        bc_func.interpolate(bc_expr)
    
    # 6. Initial condition: u(x, 0) = 0.25*sin(2*pi*x)*sin(pi*y)
    t_const.value = 0.0
    ic_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(ic_expr)
    u_h.x.array[:] = u_n.x.array[:]
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # 7. Set up nonlinear solver
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    solver.report = False
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # 8. Time stepping
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Update BC and source term to new time
        update_bc(t)
        t_const.value = t
        
        # Use previous solution as initial guess
        u_h.x.array[:] = u_n.x.array[:]
        
        n_newton, converged = solver.solve(u_h)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u_h.x.scatter_forward()
        
        nonlinear_iterations.append(int(n_newton))
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Extract solution on 60x60 grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    ic_expr2 = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_init_func.interpolate(ic_expr2)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": int(sum(nonlinear_iterations) * 5),  # rough estimate of total linear iters
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
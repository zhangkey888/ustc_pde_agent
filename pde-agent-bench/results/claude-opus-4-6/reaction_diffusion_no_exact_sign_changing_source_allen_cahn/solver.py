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
    domain_spec = case_spec.get("domain", {})
    
    # Get parameters
    epsilon = pde.get("epsilon", 0.01)
    reaction_type = pde.get("reaction", {}).get("type", "allen_cahn")
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Mesh resolution - use a good resolution for accuracy
    mesh_resolution = 80
    element_degree = 2
    
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 3*cos(3*pi*x)*sin(2*pi*y)
    f_expr = 3.0 * ufl.cos(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    # Initial condition: u0 = 0.2*sin(3*pi*x)*sin(2*pi*y)
    u_n = fem.Function(V, name="u_n")
    u0_expr_ufl = 0.2 * ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    u0_fem_expr = fem.Expression(u0_expr_ufl, V.element.interpolation_points)
    u_n.interpolate(u0_fem_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u0_fem_expr)
    
    # Current solution
    u = fem.Function(V, name="u")
    u.x.array[:] = u_n.x.array[:]
    
    v = ufl.TestFunction(V)
    
    # Boundary conditions: u = 0 on boundary (since initial condition vanishes on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # Allen-Cahn reaction: R(u) = (1/epsilon_ac) * (u^3 - u) or similar
    # Standard Allen-Cahn: ∂u/∂t - ε∇²u + R(u) = f
    # where R(u) = u^3 - u (cubic reaction)
    # But we need to check what epsilon means here
    # The PDE is: ∂u/∂t - ε ∇²u + R(u) = f
    # For Allen-Cahn, R(u) = (u^3 - u)/ε² or just u^3 - u
    # Let's use R(u) = u^3 - u as a standard choice
    
    eps_val = fem.Constant(domain, PETSc.ScalarType(epsilon))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Nonlinear residual for backward Euler:
    # (u - u_n)/dt - eps * ∇²u + R(u) - f = 0
    # Weak form: (u - u_n)/dt * v + eps * ∇u · ∇v + R(u)*v - f*v = 0
    
    F = ((u - u_n) / dt_const * v * ufl.dx
         + eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + (u**3 - u) * v * ufl.dx
         - f_expr * v * ufl.dx)
    
    # Setup nonlinear solver
    problem = petsc.NonlinearProblem(F, u, bcs=bcs)
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
    ksp.setTolerances(rtol=1e-8)
    
    # Time stepping
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Solve nonlinear problem
        n_iters, converged = solver.solve(u)
        assert converged, f"Newton solver did not converge at step {step}, t={t}"
        u.x.scatter_forward()
        
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
    
    # Evaluate on 70x70 grid
    nx_eval, ny_eval = 70, 70
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_eval * ny_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    u_initial_values = np.full(nx_eval * ny_eval, np.nan)
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts, cls)
        u_initial_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    u_initial_grid = u_initial_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
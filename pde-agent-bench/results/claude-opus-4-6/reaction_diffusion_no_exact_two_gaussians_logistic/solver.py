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
    time_spec = pde.get("time", {})
    
    # Parameters
    epsilon = pde.get("epsilon", 0.01)
    t_end = time_spec.get("t_end", 0.4)
    dt_suggested = time_spec.get("dt", 0.01)
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    
    # Mesh
    mesh_resolution = 80
    element_degree = 1
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Functions
    u = fem.Function(V, name="u")       # current solution (unknown at n+1)
    u_n = fem.Function(V, name="u_n")   # solution at time n
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = 6.0 * (ufl.exp(-160.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) + 
                     0.8 * ufl.exp(-160.0 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2)))
    
    # Initial condition
    def u0_func(X):
        return 0.3 * np.exp(-50.0 * ((X[0] - 0.3)**2 + (X[1] - 0.5)**2)) + \
               0.3 * np.exp(-50.0 * ((X[0] - 0.7)**2 + (X[1] - 0.5)**2))
    
    u_n.interpolate(u0_func)
    u.interpolate(u0_func)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u0_func)
    
    # Reaction term: logistic R(u) = u*(1-u)
    reaction_type = pde.get("reaction", {}).get("type", "logistic")
    # R(u) = u*(1-u)
    R_u = u * (1.0 - u)
    
    # Boundary conditions: u = 0 on boundary (homogeneous Dirichlet)
    bc_value = pde.get("boundary_conditions", {}).get("value", 0.0)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_value), dofs, V)
    bcs = [bc]
    
    # Time stepping with backward Euler
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Nonlinear residual: (u - u_n)/dt * v + eps * grad(u) . grad(v) + R(u) * v - f * v = 0
    F = ((u - u_n) / dt_const * v * ufl.dx +
         eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx +
         R_u * v * ufl.dx -
         f_expr * v * ufl.dx)
    
    # Setup nonlinear solver
    problem = petsc.NonlinearProblem(F, u, bcs=bcs)
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
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    for step in range(n_steps):
        n_newton, converged = solver.solve(u)
        assert converged, f"Newton solver did not converge at step {step}"
        nonlinear_iterations.append(n_newton)
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
        u.x.scatter_forward()
    
    # Evaluate on grid
    nx_grid, ny_grid = 70, 70
    xs = np.linspace(0.0, 1.0, nx_grid)
    ys = np.linspace(0.0, 1.0, ny_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_grid * ny_grid))
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
    
    u_values = np.full(nx_grid * ny_grid, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_grid, ny_grid))
    
    # Also evaluate initial condition
    u_init_values = np.full(nx_grid * ny_grid, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts, cls)
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_grid, ny_grid))
    
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
        "iterations": sum(nonlinear_iterations),  # approximate
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    # Start timing (for internal use only, not returned in solver_info)
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    # According to instruction: if Problem Description mentions t_end or dt,
    # we MUST set hardcoded defaults and force is_transient = True
    time_info = case_spec.get('pde', {}).get('time', {})
    t_end = time_info.get('t_end', 0.1)  # Default from problem description
    dt_suggested = time_info.get('dt', 0.01)  # Default from problem description
    time_scheme = time_info.get('scheme', 'backward_euler')
    
    # Coefficients
    kappa = case_spec.get('coefficients', {}).get('kappa', 1.0)
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    convergence_tol = 0.01  # 1% relative error tolerance
    
    # Initialize variables for convergence checking
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_u = None
    final_domain = None
    final_V = None
    
    # Solver info to be populated
    solver_info = {
        'mesh_resolution': None,
        'element_degree': 1,
        'ksp_type': 'gmres',
        'pc_type': 'hypre',
        'rtol': 1e-8,
        'iterations': 0,
        'dt': dt_suggested,
        'n_steps': int(t_end / dt_suggested),
        'time_scheme': time_scheme,
    }
    
    # Total linear iterations across all solves
    total_iterations = 0
    
    # Adaptive mesh loop
    for N in resolutions:
        comm = MPI.COMM_WORLD
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define trial and test functions
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step (unknown)
        v = ufl.TestFunction(V)
        
        # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
        def u0_func(x):
            return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        u_n.interpolate(u0_func)
        
        # Time-stepping parameters
        dt = dt_suggested
        n_steps = int(t_end / dt)
        if n_steps == 0:
            n_steps = 1
            dt = t_end
        
        # Update n_steps in solver_info
        solver_info['n_steps'] = n_steps
        solver_info['dt'] = dt
        
        # Time variable for source term and boundary condition
        t_var = fem.Constant(domain, ScalarType(0.0))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Source term f derived from manufactured solution
        # u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
        # f = ∂u/∂t - ∇·(κ ∇u) 
        #   = -exp(-t)*sin(pi*x)*sin(pi*y) - κ*(-2π²)*exp(-t)*sin(pi*x)*sin(pi*y)
        #   = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2κπ²)
        # With κ=1: f = exp(-t)*sin(pi*x)*sin(pi*y)*(2π² - 1)
        f_expr = ufl.exp(-t_var) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * (2 * np.pi**2 * kappa - 1)
        
        # Variational form for backward Euler (linear in u)
        # (u - u_n)/dt * v * dx + κ * inner(grad(u), grad(v)) * dx = f * v * dx
        # Rearranged as: a(u, v) = L(v)
        a = (1.0/dt) * ufl.inner(u, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1.0/dt) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
        
        # Boundary condition: u = g on ∂Ω
        # g = exp(-t)*sin(pi*x)*sin(pi*y) on boundary
        def boundary_marker(x):
            # Mark all boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        
        # Time-stepping loop
        linear_iterations_this_mesh = 0
        
        for step in range(n_steps):
            t = (step + 1) * dt
            t_var.value = t
            
            # Update boundary condition for current time
            def bc_func(x):
                return np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
            u_bc.interpolate(bc_func)
            
            # Apply boundary condition
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Try iterative solver first
            try:
                petsc_options = {
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                }
                
                problem = petsc.LinearProblem(
                    a, L, 
                    bcs=[bc], 
                    u=u,
                    petsc_options_prefix='pdebench_',
                    petsc_options=petsc_options
                )
                
                # Solve
                u_sol = problem.solve()
                
                # Get KSP information
                ksp = problem.solver
                linear_its = ksp.getIterationNumber()
                linear_iterations_this_mesh += linear_its
                
                # Update solver info based on what worked
                if step == 0:  # Only set once per mesh
                    ksp_type = ksp.getType()
                    pc = ksp.getPC()
                    pc_type = pc.getType()
                    solver_info['ksp_type'] = str(ksp_type).split('.')[-1].lower()
                    solver_info['pc_type'] = str(pc_type).split('.')[-1].lower()
                
            except Exception as e:
                # If iterative solver fails, try with direct solver
                print(f"Iterative solver failed at step {step}: {e}, trying direct solver")
                petsc_options = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                }
                
                problem = petsc.LinearProblem(
                    a, L, 
                    bcs=[bc], 
                    u=u,
                    petsc_options_prefix='pdebench_',
                    petsc_options=petsc_options
                )
                
                u_sol = problem.solve()
                
                # Get KSP information
                ksp = problem.solver
                linear_its = ksp.getIterationNumber()
                linear_iterations_this_mesh += linear_its
                
                # Update solver info
                if step == 0:  # Only set once
                    ksp_type = ksp.getType()
                    pc = ksp.getPC()
                    pc_type = pc.getType()
                    solver_info['ksp_type'] = str(ksp_type).split('.')[-1].lower()
                    solver_info['pc_type'] = str(pc_type).split('.')[-1].lower()
            
            # Update for next time step
            u_n.x.array[:] = u.x.array[:]
        
        total_iterations += linear_iterations_this_mesh
        
        # Compute L2 norm of solution for convergence checking
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if relative_error < convergence_tol:
                # Converged!
                final_solution = u
                final_mesh_resolution = N
                final_u = u
                final_domain = domain
                final_V = V
                solver_info['mesh_resolution'] = N
                solver_info['iterations'] = total_iterations
                break
        
        prev_norm = norm_value
        final_solution = u
        final_mesh_resolution = N
        final_u = u
        final_domain = domain
        final_V = V
    
    # If loop finished without break, use the finest mesh result
    if solver_info['mesh_resolution'] is None:
        solver_info['mesh_resolution'] = resolutions[-1]
        solver_info['iterations'] = total_iterations
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also get initial condition on the same grid
    u0_sampled = fem.Function(final_V)
    def u0_exact(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    u0_sampled.interpolate(u0_exact)
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_sampled.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    u0_grid = u0_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

# For testing if run directly
if __name__ == "__main__":
    # Test case
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        },
        "coefficients": {
            "kappa": 1.0
        }
    }
    
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Total linear iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    
    # Compute error against exact solution
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Exact solution at t=0.1: exp(-0.1)*sin(pi*x)*sin(pi*y)
    u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.abs(result['u'] - u_exact)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    print(f"Accuracy requirement: ≤ 1.42e-03")
    print(f"Pass accuracy: {max_error <= 1.42e-03}")


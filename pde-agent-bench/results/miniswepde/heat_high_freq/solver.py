import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time
import math

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    t_end = 0.1
    dt_suggested = 0.005
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # For safety, ensure we have time parameters
    if t_end is None:
        t_end = 0.1
    if dt_suggested is None:
        dt_suggested = 0.005
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    
    # Initialize solver info
    solver_info = {
        "mesh_resolution": None,
        "element_degree": 1,
        "ksp_type": None,
        "pc_type": None,
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt_suggested,
        "n_steps": int(t_end / dt_suggested),
        "time_scheme": scheme,
        "nonlinear_iterations": []
    }
    
    # Grid convergence loop
    for i, N in enumerate(resolutions):
        comm = MPI.COMM_WORLD
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Time parameters
        dt = dt_suggested
        n_steps = int(t_end / dt)
        if n_steps == 0:
            n_steps = 1
        if n_steps * dt < t_end:
            n_steps += 1  # Ensure we reach t_end
        
        # Define initial condition
        u_n = fem.Function(V)
        # u(x,0) = sin(4*pi*x)*sin(4*pi*y)
        def u0_expr(x):
            return np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
        u_n.interpolate(u0_expr)
        
        # Define boundary condition (Dirichlet)
        # Use exact solution at boundaries
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
        total_iterations = 0
        current_time = 0.0
        
        # Store initial condition for output
        u_initial = u_n.x.array.copy()
        
        # Try iterative solver first
        solver_choice = "iterative"
        ksp_type = "gmres"
        pc_type = "hypre"
        
        for step in range(n_steps):
            current_time = (step + 1) * dt  # Time at n+1 for backward Euler
            if current_time > t_end:
                current_time = t_end
            
            # Update boundary condition for current time
            def bc_expr_t(x):
                t_val = current_time
                return np.exp(-t_val) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
            u_bc.interpolate(bc_expr_t)
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Define source term f at time t_{n+1}
            # f = (32*pi^2 - 1) * u_exact
            # where u_exact = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
            
            # Create a function for f
            f_func = fem.Function(V)
            def f_expr(x):
                t_val = current_time
                u_exact_val = np.exp(-t_val) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
                return (32 * math.pi**2 - 1) * u_exact_val
            f_func.interpolate(f_expr)
            
            # Variational form for backward Euler:
            # ∫ u^{n+1} v dx + Δt ∫ κ∇u^{n+1}·∇v dx = ∫ u^n v dx + Δt ∫ f^{n+1} v dx
            
            # Left-hand side
            a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            
            # Right-hand side
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
            
            # Try to solve with LinearProblem
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": 1e-8},
                    petsc_options_prefix="heat_"
                )
                u_sol = problem.solve()
                
                # Get iteration count if available
                if hasattr(problem.solver, 'getIterationNumber'):
                    total_iterations += problem.solver.getIterationNumber()
                
            except Exception as e:
                # Fallback to direct solver
                solver_choice = "direct"
                ksp_type = "preonly"
                pc_type = "lu"
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
                    petsc_options_prefix="heat_"
                )
                u_sol = problem.solve()
                
                if hasattr(problem.solver, 'getIterationNumber'):
                    total_iterations += problem.solver.getIterationNumber()
            
            # Update u_n for next step
            u_n.x.array[:] = u_sol.x.array
        
        # Final solution at t_end
        final_solution = u_n
        
        # Compute norm of solution
        norm_form = fem.form(ufl.inner(final_solution, final_solution) * ufl.dx)
        norm = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm)
        solutions.append((final_solution, domain, u_initial))
        
        # Update solver info
        solver_info["ksp_type"] = ksp_type
        solver_info["pc_type"] = pc_type
        solver_info["iterations"] = total_iterations
        
        # Check convergence
        if i > 0:
            relative_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] != 0 else float('inf')
            if relative_error < 0.01:  # 1% convergence criterion
                solver_info["mesh_resolution"] = N
                break
        
        # If we reach the last resolution, use it
        if i == len(resolutions) - 1:
            solver_info["mesh_resolution"] = N
    
    # Get the converged solution (or last one if no convergence)
    if solutions:
        u_sol, domain, u_initial = solutions[-1]
    else:
        # Fallback
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, 64, 64, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        u_sol = fem.Function(V)
        def sol_expr(x):
            return np.exp(-t_end) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
        u_sol.interpolate(sol_expr)
        
        u_initial_func = fem.Function(V)
        def init_expr(x):
            return np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
        u_initial_func.interpolate(init_expr)
        u_initial = u_initial_func.x.array.copy()
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D coordinates for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also sample initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_initial
    u_initial_vals = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_vals.reshape((nx, ny))
    
    # End timing
    end_time = time.time()
    solver_info["wall_time_sec"] = end_time - start_time
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a simple case specification
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.005,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    print(f"Time taken: {result['solver_info']['wall_time_sec']:.2f} seconds")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve transient heat equation with adaptive mesh refinement and runtime auto-tuning.
    
    Args:
        case_spec: Dictionary containing PDE specification. For heat equation,
                   should contain 'pde' -> 'time' -> 't_end', 'dt', 'scheme'.
                   If missing, uses hardcoded defaults from problem description.
    
    Returns:
        Dictionary with keys:
        - "u": final solution on 50x50 grid (numpy array)
        - "u_initial": initial condition on same grid
        - "solver_info": dictionary with solver parameters and performance metrics
    """
    # Start timing
    start_time = time.time()
    
    # Hardcoded defaults from problem description
    # Problem Description states: t_end = 0.08, dt (suggested) = 0.008
    DEFAULT_T_END = 0.08
    DEFAULT_DT = 0.008
    DEFAULT_SCHEME = 'backward_euler'
    
    # Extract parameters from case_spec with hardcoded defaults
    time_params = case_spec.get('pde', {}).get('time', {})
    t_end = time_params.get('t_end', DEFAULT_T_END)
    dt_suggested = time_params.get('dt', DEFAULT_DT)
    scheme = time_params.get('scheme', DEFAULT_SCHEME)
    
    # Force transient treatment (heat equation is always transient)
    is_transient = True
    
    # Manufactured solution
    def exact_solution(x, t):
        """u_exact = exp(-t)*sin(pi*x)*sin(2*pi*y)"""
        return np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Source term f = ∂u/∂t - ∇·(κ ∇u) where κ=1.0
    # f = exp(-t)*sin(pi*x)*sin(2*pi*y)*(5*pi^2 - 1)
    def source_term(x, t):
        return (5 * np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    errors = []  # Store L2 errors against exact solution
    mesh_resolution_used = None
    element_degree = 1  # Linear elements
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Solver configuration history
    ksp_type_used = None
    pc_type_used = None
    rtol_used = None
    linear_iterations_total = 0
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition: Dirichlet using exact solution
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # All boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Time-stepping parameters
        dt = dt_suggested
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # Adjust dt to exactly reach t_end
        
        # Functions for current and previous time steps
        u_n = fem.Function(V)  # u at previous time step
        
        # Set initial condition u_n(x, 0) = exact_solution(x, 0)
        u_n.interpolate(lambda x: exact_solution(x, 0.0))
        
        # Trial and test functions
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)
        
        # Time-stepping loop
        step_iterations = 0
        
        for step in range(n_steps):
            t = (step + 1) * dt
            
            # Update boundary condition with exact solution at time t
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: exact_solution(x, t))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Create source function at time t
            f_func = fem.Function(V)
            f_func.interpolate(lambda x: source_term(x, t))
            
            # Backward Euler weak form: (u, v) + dt*(grad(u), grad(v)) = (u_n, v) + dt*(f, v)
            dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
            a = ufl.inner(u, v) * ufl.dx + dt_constant * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt_constant * ufl.inner(f_func, v) * ufl.dx
            
            # Try different solver configurations (only on first step to choose)
            if step == 0:
                solver_configs = [
                    {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
                    {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-8},
                    {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
                ]
            else:
                # Use the same config that worked on first step
                solver_configs = [{"ksp_type": ksp_type_used, "pc_type": pc_type_used, "rtol": rtol_used}]
            
            solver_success = False
            for solver_config in solver_configs:
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={
                            "ksp_type": solver_config["ksp_type"],
                            "pc_type": solver_config["pc_type"],
                            "ksp_rtol": solver_config["rtol"]
                        },
                        petsc_options_prefix=f"heat_step{step}_"
                    )
                    u_sol = problem.solve()
                    
                    # Get iteration count
                    iterations = problem.solver.getIterationNumber()
                    step_iterations += iterations
                    
                    # Store solver config on first successful step
                    if step == 0 and not solver_success:
                        ksp_type_used = solver_config["ksp_type"]
                        pc_type_used = solver_config["pc_type"]
                        rtol_used = solver_config["rtol"]
                    
                    solver_success = True
                    break
                except Exception as e:
                    if rank == 0 and step == 0:
                        print(f"  Solver config {solver_config} failed: {e}")
                    continue
            
            if not solver_success:
                raise RuntimeError(f"All solver configurations failed at step {step}")
            
            # Update for next step
            u_n.x.array[:] = u_sol.x.array
        
        # Accumulate total iterations
        linear_iterations_total += step_iterations
        
        # After time-stepping, u_n contains final solution
        u_final = u_n
        
        # Compute L2 error against exact solution at final time
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: exact_solution(x, t_end))
        
        error_func = fem.Function(V)
        error_func.x.array[:] = u_final.x.array - u_exact.x.array
        error_norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(error_func, error_func) * ufl.dx)))
        errors.append(error_norm)
        solutions.append((u_final, domain))
        
        if rank == 0:
            print(f"  L2 error at N={N}: {error_norm:.6e}")
            print(f"  Linear iterations for this resolution: {step_iterations}")
        
        # Check if error meets requirement (7.97e-04 from problem description)
        if error_norm < 7.97e-04:
            mesh_resolution_used = N
            if rank == 0:
                print(f"  ✓ Accuracy requirement met at N={N} with error {error_norm:.6e}")
            break
    
    # If loop finished without meeting accuracy, use finest mesh
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
        if rank == 0:
            print(f"  Using finest mesh resolution N={mesh_resolution_used}")
            print(f"  Final error: {errors[-1]:.6e}")
    
    # Get the final solution from the chosen resolution
    final_idx = resolutions.index(mesh_resolution_used)
    u_final, final_domain = solutions[final_idx]
    
    # Sample solution on 50x50 grid (as required by evaluator)
    nx, ny = 50, 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points array (3D for geometry utilities)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    if len(points_on_proc) > 0:
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values on root process
    u_values_all = comm.gather(u_values, root=0)
    
    # Get initial condition for output (recommended for time-dependent PDEs)
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: exact_solution(x, 0.0))
    
    u_initial_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    if len(points_on_proc) > 0:
        vals_initial = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_values[eval_map] = vals_initial.flatten()
    
    u_initial_all = comm.gather(u_initial_values, root=0)
    
    if rank == 0:
        # Combine results from all processes for final solution
        u_combined = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
        for proc_vals in u_values_all:
            mask = ~np.isnan(proc_vals)
            u_combined[mask] = proc_vals[mask]
        
        # Reshape to 50x50 grid
        u_grid = u_combined.reshape((nx, ny))
        
        # Combine results for initial condition
        u_initial_combined = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
        for proc_vals in u_initial_all:
            mask = ~np.isnan(proc_vals)
            u_initial_combined[mask] = proc_vals[mask]
        
        u_initial_grid = u_initial_combined.reshape((nx, ny))
    else:
        u_grid = None
        u_initial_grid = None
    
    # Broadcast results from root to all processes
    u_grid = comm.bcast(u_grid, root=0)
    u_initial_grid = comm.bcast(u_initial_grid, root=0)
    
    # Prepare solver info (must match required format)
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": linear_iterations_total,  # total linear solver iterations
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    end_time = time.time()
    if rank == 0:
        print(f"\n=== Solver Summary ===")
        print(f"Total solve time: {end_time - start_time:.2f} seconds")
        print(f"Final L2 error: {errors[final_idx]:.6e}")
        print(f"Accuracy requirement: ≤ 7.97e-04")
        print(f"Time limit: ≤ 49.053 seconds")
        print(f"Mesh resolution used: {mesh_resolution_used}")
        print(f"Time step dt: {dt}")
        print(f"Number of time steps: {n_steps}")
        print(f"Total linear iterations: {linear_iterations_total}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a minimal case specification
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.008,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("\nSolver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Final solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

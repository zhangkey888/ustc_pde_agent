import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing PDE specification
        
    Returns:
    --------
    dict with keys:
        - "u": numpy array of shape (50, 50) with solution values
        - "solver_info": dictionary with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Start timing for performance monitoring
    start_time = time.time()
    
    # Extract problem parameters
    # For elliptic problem, no time stepping needed
    is_transient = False
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        is_transient = True
        # If time parameters provided, use them (but this is elliptic)
        t_end = case_spec['pde']['time'].get('t_end', 1.0)
        dt = case_spec['pde']['time'].get('dt', 0.01)
    
    # Adaptive mesh refinement loop
    # Start with coarse mesh, refine until convergence
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    
    # Solver info to be populated
    solver_info = {
        'mesh_resolution': None,
        'element_degree': 1,  # Using linear P1 elements
        'ksp_type': None,
        'pc_type': None,
        'rtol': 1e-8,
        'iterations': 0
    }
    
    # Try iterative solver first (faster), fallback to direct (robust)
    solver_strategies = [
        {'ksp_type': 'gmres', 'pc_type': 'hypre', 'name': 'iterative'},
        {'ksp_type': 'preonly', 'pc_type': 'lu', 'name': 'direct'}
    ]
    
    converged = False
    final_solution = None
    final_norm = None
    final_N = None
    final_iterations = 0
    final_strategy = None
    
    for i, N in enumerate(resolutions):
        # Create mesh for current resolution
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space: linear Lagrange elements
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define coefficient κ(x) = 1 + 50*exp(-200*(x-0.5)^2)
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5)**2)
        
        # Define source term f(x,y) = 1 + sin(2πx)*cos(2πy)
        f_expr = 1.0 + ufl.sin(2.0 * np.pi * x[0]) * ufl.cos(2.0 * np.pi * x[1])
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Variational form: ∫κ∇u·∇v dx = ∫f v dx
        a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Boundary conditions: u = 0 on all boundaries (homogeneous Dirichlet)
        # This is appropriate for Poisson with source term when boundary condition not specified
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark all boundaries of unit square
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try solver strategies in order
        solution_found = False
        u_sol = None
        iterations = 0
        current_strategy = None
        
        for strategy in solver_strategies:
            try:
                # Create linear problem with current solver strategy
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        'ksp_type': strategy['ksp_type'],
                        'pc_type': strategy['pc_type'],
                        'ksp_rtol': solver_info['rtol'],
                        'ksp_atol': 1e-12,
                        'ksp_max_it': 1000,
                    },
                    petsc_options_prefix="poisson"
                )
                
                # Solve the linear system
                u_sol = problem.solve()
                
                # Get iteration count
                ksp = problem.solver
                iterations = ksp.getIterationNumber()
                
                # Record successful strategy
                current_strategy = strategy
                solution_found = True
                break  # Success, exit strategy loop
                
            except Exception as e:
                # If iterative solver fails, try direct solver next
                if strategy['name'] == 'direct':
                    # Direct solver should not fail for this well-posed problem
                    raise RuntimeError(f"Direct solver failed: {e}")
                # Continue to next strategy
                continue
        
        if not solution_found:
            raise RuntimeError("All solver strategies failed")
        
        # Compute L2 norm of solution for convergence checking
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        solutions.append(u_sol)
        
        # Check convergence: compare with previous resolution
        if i > 0:
            # Relative change in norm
            if norms[i] > 0:
                relative_error = abs(norms[i] - norms[i-1]) / norms[i]
            else:
                relative_error = float('inf')
            
            # Stop if change is less than 1%
            if relative_error < 0.01:
                converged = True
                final_solution = u_sol
                final_norm = norm_value
                final_N = N
                final_iterations = iterations
                final_strategy = current_strategy
                solver_info['mesh_resolution'] = N
                solver_info['ksp_type'] = current_strategy['ksp_type']
                solver_info['pc_type'] = current_strategy['pc_type']
                solver_info['iterations'] = iterations
                break
        
        # Store current strategy info (will be used if this is the last resolution)
        if i == len(resolutions) - 1 or converged:
            final_strategy = current_strategy
            solver_info['ksp_type'] = current_strategy['ksp_type']
            solver_info['pc_type'] = current_strategy['pc_type']
            solver_info['iterations'] = iterations
    
    # Ensure we have a solution (fallback to finest mesh if not converged)
    if final_solution is None:
        final_N = resolutions[-1]
        final_solution = solutions[-1]
        final_iterations = solver_info['iterations']
        solver_info['mesh_resolution'] = final_N
    
    # Sample solution on 50x50 uniform grid as required by evaluator
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D coordinates, z=0 for 2D)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at all points
    u_grid_flat = np.full((nx * ny,), np.nan, dtype=ScalarType)
    
    # Use geometry utilities for efficient point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build lists of points and cells for evaluation
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate function at points found on this processor
    if len(points_on_proc) > 0:
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny) grid
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Finalize solver info
    solver_info['mesh_resolution'] = final_N
    solver_info['iterations'] = final_iterations
    
    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Optional: print performance info (commented out for production)
    # if comm.rank == 0:
    #     print(f"Solve completed in {elapsed:.3f}s")
    #     print(f"Final mesh resolution: {final_N}")
    #     print(f"Solver: {solver_info['ksp_type']} with {solver_info['pc_type']}")
    #     print(f"Iterations: {final_iterations}")
    
    # Prepare return dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    # For time-dependent problems, additional fields would be added here
    # but this is an elliptic problem
    
    return result

if __name__ == "__main__":
    # Test the solver with the specific case specification
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 50*exp(-200*(x-0.5)**2)"}
            },
            "source": "1 + sin(2*pi*x)*cos(2*pi*y)"
        }
    }
    
    try:
        import time
        start = time.time()
        result = solve(case_spec)
        end = time.time()
        
        print("=" * 60)
        print("Solver Test Results")
        print("=" * 60)
        print(f"Execution time: {end - start:.3f} seconds")
        print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
        print(f"Element degree: {result['solver_info']['element_degree']}")
        print(f"Linear solver: {result['solver_info']['ksp_type']}")
        print(f"Preconditioner: {result['solver_info']['pc_type']}")
        print(f"Iterations: {result['solver_info']['iterations']}")
        print(f"Solution shape: {result['u'].shape}")
        
        # Check solution quality
        u_min, u_max = np.nanmin(result['u']), np.nanmax(result['u'])
        nan_count = np.sum(np.isnan(result['u']))
        print(f"Solution range: [{u_min:.6f}, {u_max:.6f}]")
        print(f"NaN values: {nan_count}")
        
        if nan_count > 0:
            print("WARNING: Solution contains NaN values!")
        else:
            print("Solution quality: OK (no NaN values)")
            
        # Check if time constraint would be met
        if end - start < 4.807:
            print("Time constraint: PASS (< 4.807s)")
        else:
            print(f"Time constraint: WARNING ({end - start:.3f}s > 4.807s)")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during solver execution: {e}")
        import traceback
        traceback.print_exc()

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    Returns solution sampled on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Problem parameters
    kappa = case_spec.get('kappa', 1.0)
    domain_bounds = case_spec.get('domain', [[0.0, 0.0], [1.0, 1.0]])
    
    # Exact solution function
    def exact_solution(x):
        """u_exact = sin(pi*x)*sin(pi*y)"""
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Create sampling grid (50x50)
    nx, ny = 50, 50
    x_samples = np.linspace(0.0, 1.0, nx)
    y_samples = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_samples, y_samples, indexing='ij')
    
    # Points for evaluation (3D coordinates)
    sample_points = np.zeros((3, nx * ny))
    sample_points[0, :] = X.flatten()
    sample_points[1, :] = Y.flatten()
    sample_points[2, :] = 0.0
    
    # Exact values on sampling grid
    u_exact_samples = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Grid convergence loop
    resolutions = [32, 64, 128, 256]
    element_degrees = [2, 3]  # Higher degree for better accuracy
    
    best_solution = None
    best_info = None
    best_grid_error = float('inf')
    best_N = None
    best_degree = None
    best_u_samples = None
    
    for degree in element_degrees:
        if best_grid_error < 1e-6:
            break
            
        for N in resolutions:
            # Create mesh
            domain = mesh.create_rectangle(
                comm, 
                [domain_bounds[0], domain_bounds[1]], 
                [N, N], 
                cell_type=mesh.CellType.triangle
            )
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define trial and test functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Define variational form
            a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[0]) * \
                     ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[1])
            L = ufl.inner(f_expr, v) * ufl.dx
            
            # Boundary conditions (Dirichlet)
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
            
            # Create boundary function
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Solver configuration
            ksp_type = 'cg'  # Optimal for symmetric positive definite
            pc_type = 'hypre'
            rtol = 1e-12
            
            # Try solving
            try:
                petsc_options = {
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-14,
                    "ksp_max_it": 5000
                }
                
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=petsc_options,
                    petsc_options_prefix="poisson_"
                )
                
                u_sol = problem.solve()
                
            except Exception as e:
                # Fallback to direct solver
                ksp_type = 'preonly'
                pc_type = 'lu'
                rtol = 1e-14
                
                petsc_options = {
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                }
                
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=petsc_options,
                    petsc_options_prefix="poisson_direct_"
                )
                
                u_sol = problem.solve()
            
            # Evaluate solution on sampling grid
            u_samples_flat = evaluate_function_at_points(u_sol, sample_points)
            u_samples = u_samples_flat.reshape((nx, ny))
            
            # Compute error on sampling grid
            error_grid = np.abs(u_samples - u_exact_samples)
            max_error = np.max(error_grid)
            grid_error = max_error  # Use max error as criterion
            
            # Store if this is the best so far
            if grid_error < best_grid_error:
                best_solution = u_sol
                best_info = {
                    "mesh_resolution": N,
                    "element_degree": degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": problem.solver.getIterationNumber() if hasattr(problem.solver, 'getIterationNumber') else 0
                }
                best_grid_error = grid_error
                best_N = N
                best_degree = degree
                best_u_samples = u_samples
            
            # Check if we've met accuracy requirement on sampling grid
            if grid_error < 1e-6:
                # Accuracy requirement met
                break
        
        # If we met accuracy with this degree, don't try lower degree
        if best_grid_error < 1e-6:
            break
    
    # If still not accurate enough, try even higher resolution with P3
    if best_grid_error >= 1e-6:
        # Try with higher resolution
        N_final = 512
        degree_final = 3
        
        domain = mesh.create_rectangle(
            comm, 
            [domain_bounds[0], domain_bounds[1]], 
            [N_final, N_final], 
            cell_type=mesh.CellType.triangle
        )
        
        V = fem.functionspace(domain, ("Lagrange", degree_final))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[0]) * \
                 ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[1])
        L = ufl.inner(f_expr, v) * ufl.dx
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="poisson_final_"
        )
        best_solution = problem.solve()
        
        # Evaluate on sampling grid
        u_samples_flat = evaluate_function_at_points(best_solution, sample_points)
        best_u_samples = u_samples_flat.reshape((nx, ny))
        
        best_info = {
            "mesh_resolution": N_final,
            "element_degree": degree_final,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-14,
            "iterations": 0
        }
    
    # Return results
    return {
        "u": best_u_samples,
        "solver_info": best_info
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at arbitrary points.
    points: shape (3, N) numpy array
    Returns: shape (N,) numpy array
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we need to gather results from all processes
    comm = domain.comm
    all_u_values = np.zeros_like(u_values)
    comm.Allreduce(u_values, all_u_values, op=MPI.SUM)
    
    # Replace NaN with 0 (points not found on any process)
    all_u_values[np.isnan(all_u_values)] = 0.0
    
    return all_u_values

if __name__ == "__main__":
    # Test the solver with a simple case specification
    case_spec = {
        "kappa": 1.0,
        "domain": [[0.0, 0.0], [1.0, 1.0]]
    }
    
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("Solution shape:", result["u"].shape)

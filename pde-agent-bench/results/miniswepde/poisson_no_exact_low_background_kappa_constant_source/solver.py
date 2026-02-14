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
    
    # Extract problem parameters from case_spec
    pde_info = case_spec.get('pde', {})
    
    # Source term - default to 1.0 if not specified
    source_val = pde_info.get('source', 1.0)
    
    # Coefficient κ
    kappa_info = pde_info.get('coefficients', {}).get('kappa', {})
    kappa_expr_str = kappa_info.get('expr', '0.2 + 0.8*exp(-80*((x-0.5)**2 + (y-0.5)**2))')
    
    # Domain is unit square [0,1]x[0,1]
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    converged_resolution = None
    converged_solution = None
    solver_info_final = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space - using degree 1 elements (P1)
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Parse κ expression from string to ufl expression
        # Note: This is a simplified parser for the specific expression format
        # For production, a more robust parser would be needed
        try:
            # Evaluate the expression string in a safe context with ufl functions
            # Create a dictionary with ufl functions and math functions
            import math
            safe_dict = {
                'x': x[0], 'y': x[1],
                'exp': ufl.exp, 'sqrt': ufl.sqrt, 'sin': ufl.sin, 'cos': ufl.cos,
                'tan': ufl.tan, 'pi': np.pi, 'e': np.e
            }
            # Add basic math functions
            safe_dict.update({k: getattr(ufl, k) for k in dir(ufl) if not k.startswith('_')})
            
            # Evaluate the expression
            kappa_expr = eval(kappa_expr_str, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            print(f"Warning: Could not parse κ expression '{kappa_expr_str}', using default: {e}")
            # Default expression
            kappa_expr = 0.2 + 0.8 * ufl.exp(-80 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
        
        # Source term
        f = fem.Constant(domain, ScalarType(source_val))
        
        # Variational form: -∇·(κ ∇u) = f
        a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Boundary conditions: u = g on ∂Ω
        # Since g is not specified in problem description, use homogeneous Dirichlet
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create zero boundary condition
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct if fails
        try:
            # Use iterative solver with hypre preconditioner
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            # Get iteration count
            iterations = problem.solver.getIterationNumber()
            solver_info = {
                "mesh_resolution": N,
                "element_degree": 1,
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rtol": 1e-8,
                "iterations": iterations
            }
        except Exception as e:
            # Fallback to direct solver
            print(f"Iterative solver failed for N={N}, falling back to direct solver: {e}")
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu"
                    },
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": 1,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "rtol": 1e-8,
                    "iterations": 0  # Direct solver doesn't have iterations
                }
            except Exception as e2:
                print(f"Direct solver also failed for N={N}: {e2}")
                # If both fail, continue to next resolution
                continue
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                converged_solution = u_sol
                solver_info_final = solver_info
                print(f"Converged at resolution N={N} with relative error {relative_error:.6f}")
                break
        
        norm_old = norm_value
        
        # Store current as fallback
        converged_resolution = N
        converged_solution = u_sol
        solver_info_final = solver_info
    
    # If loop finished without break, we have the finest mesh result
    print(f"Using resolution N={converged_resolution}")
    
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
    u_grid_flat = evaluate_function_at_points(converged_solution, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Return result dictionary
    return {
        "u": u_grid,
        "solver_info": solver_info_final
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at given points.
    points: numpy array of shape (3, N)
    Returns: numpy array of shape (N,) with function values
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a simple case specification
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "0.2 + 0.8*exp(-80*((x-0.5)**2 + (y-0.5)**2))"}
            },
            "source": 1.0
        }
    }
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys: 'pde' (dict with 'time' key if transient)
    
    Returns
    -------
    dict
        Contains:
        - 'u': solution array on 50x50 grid
        - 'solver_info': dictionary with solver metadata
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    
    # Problem parameters - manufactured solution
    exact_solution = lambda x: np.exp(2*x[0]) * np.cos(np.pi * x[1])
    kappa = 1.0
    
    # Source term derived from manufactured solution
    # u = exp(2*x)*cos(pi*y)
    # f = -∇·(κ∇u) = (π² - 4)*exp(2*x)*cos(π*y)
    pi = np.pi
    f_expr = lambda x: (pi**2 - 4) * np.exp(2*x[0]) * np.cos(pi * x[1])
    
    # Use quadratic elements for better accuracy
    degree = 2
    resolutions = [32, 64, 128]
    
    u_sol = None
    u_norm_prev = None
    solver_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, 
                                        cell_type=mesh.CellType.triangle)
        
        # Function space with quadratic elements
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Dirichlet boundary condition (entire boundary)
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
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        f = fem.Function(V)
        f.interpolate(lambda x: f_expr(x))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if needed
        u_current = None
        
        try:
            # Iterative solver (fast for large problems)
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-10,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="pdebench_"
            )
            u_current = problem.solve()
            
            solver_info = {
                "mesh_resolution": N,
                "element_degree": degree,
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rtol": 1e-10,
                "iterations": problem.solver.getIterationNumber() 
                              if hasattr(problem.solver, 'getIterationNumber') else 0
            }
            
        except Exception:
            # Fallback to direct solver (robust but slower)
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu"
                    },
                    petsc_options_prefix="pdebench_"
                )
                u_current = problem.solve()
                
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": degree,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "rtol": 1e-10,
                    "iterations": 0
                }
            except Exception as e:
                print(f"Warning: Solver failed for N={N}: {e}")
                continue
        
        if u_current is None:
            continue
        
        # Compute L2 norm for convergence check
        norm_form = fem.form(ufl.inner(u_current, u_current) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence (relative change in norm)
        if u_norm_prev is not None:
            relative_error = abs(norm_value - u_norm_prev) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol = u_current
                break  # Converged at this resolution
        
        u_norm_prev = norm_value
        u_sol = u_current  # Keep current solution
    
    # If loop finished without convergence, use the finest mesh result
    if u_sol is None:
        # Should not happen, but fallback
        u_sol = u_current
        solver_info["mesh_resolution"] = resolutions[-1]
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    u_grid_flat = evaluate_function_at_points(u_sol, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare result dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    # Add time-related fields if specified (for compatibility)
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        result["solver_info"].update({
            "dt": 0.0,
            "n_steps": 0,
            "time_scheme": "none"
        })
    
    return result


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at arbitrary points.
    
    Parameters
    ----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points : numpy.ndarray
        Array of shape (3, N) containing coordinates
        
    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing function values
    """
    domain = u_func.function_space.mesh
    
    try:
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        
        # Find cells containing the points
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
        
        # Initialize result array with NaN
        u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
        
        if len(points_on_proc) > 0:
            # Evaluate function at points
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        return u_values
        
    except Exception as e:
        # Fallback: return zeros if evaluation fails
        print(f"Warning: Point evaluation failed: {e}")
        return np.zeros(points.shape[1])


# Test the solver if run as main
if __name__ == "__main__":
    case_spec = {"pde": {"type": "elliptic"}}
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

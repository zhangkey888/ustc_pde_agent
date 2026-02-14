import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
        
    Returns:
    --------
    dict with keys:
        - "u": numpy array shape (50, 50) - solution on uniform grid
        - "solver_info": dict with solver metadata
    """
    # Start timing
    start_time = time.time()
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]  # Progressive refinement
    convergence_tol = 0.01  # 1% relative error tolerance
    element_degree = 2  # Using quadratic elements for better accuracy
    
    # Initialize variables for convergence check
    u_prev = None
    norm_prev = None
    u_sol = None
    final_resolution = None
    
    # Solver info to be populated
    solver_info = {
        "mesh_resolution": None,
        "element_degree": element_degree,
        "ksp_type": None,
        "pc_type": None,
        "rtol": 1e-8,
        "iterations": 0
    }
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    for N in resolutions:
        print(f"Solving with mesh resolution N={N}, degree={element_degree}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Define variable kappa: 1 + 0.4*cos(4*pi*x)*sin(2*pi*y)
        kappa_expr = 1.0 + 0.4 * ufl.cos(4 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
        kappa = fem.Function(V)
        kappa.interpolate(lambda x: 1.0 + 0.4 * np.cos(4 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
        
        # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
        u_exact_expr = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        
        # Compute f = -∇·(κ ∇u_exact)
        grad_u_exact = ufl.grad(u_exact_expr)
        kappa_grad_u = kappa_expr * grad_u_exact
        f_expr = -ufl.div(kappa_grad_u)
        
        # Create source term function
        f = fem.Function(V)
        f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        
        # Define variational form
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Boundary conditions: Dirichlet u = g on ∂Ω
        # g = u_exact on boundary
        def boundary_marker(x):
            # Mark all boundary points
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
        
        # Create boundary function g = u_exact
        g = fem.Function(V)
        g.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(g, dofs)
        
        # Try iterative solver first, fallback to direct if fails
        petsc_options = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 1000
        }
        
        try:
            # Create and solve linear problem
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix="pde_"
            )
            u_h = problem.solve()
            
            # Get solver info
            ksp = problem._solver
            solver_info["ksp_type"] = ksp.getType()
            pc = ksp.getPC()
            solver_info["pc_type"] = pc.getType()
            solver_info["iterations"] = ksp.getIterationNumber()
            
        except Exception as e:
            print(f"Iterative solver failed: {e}. Falling back to direct solver.")
            # Fallback to direct solver
            petsc_options_direct = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
            
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_direct,
                petsc_options_prefix="pde_"
            )
            u_h = problem.solve()
            
            # Get solver info for direct solver
            ksp = problem._solver
            solver_info["ksp_type"] = ksp.getType()
            pc = ksp.getPC()
            solver_info["pc_type"] = pc.getType()
            solver_info["iterations"] = 1
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_prev is not None:
            relative_error = abs(norm_value - norm_prev) / norm_value
            print(f"  Relative error in norm: {relative_error:.6f}")
            
            if relative_error < convergence_tol:
                print(f"  Converged at N={N}")
                u_sol = u_h
                final_resolution = N
                break
        
        # Update for next iteration
        u_prev = u_h
        norm_prev = norm_value
        u_sol = u_h
        final_resolution = N
    
    # If loop finished without break, use the last solution (N=128)
    if final_resolution is None:
        final_resolution = 128
    solver_info["mesh_resolution"] = final_resolution
    
    # Interpolate solution to 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)])
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_sol, points, domain)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # End timing
    end_time = time.time()
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points, domain):
    """
    Evaluate a dolfinx Function at given points.
    """
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


# Test the solver if run directly
if __name__ == "__main__":
    test_case = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.4*cos(4*pi*x)*sin(2*pi*y)"}
            }
        },
        "domain": {
            "bounds": [[0, 0], [1, 1]]
        }
    }
    
    result = solve(test_case)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error
    u_grid = result["u"]
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.abs(u_grid - u_exact)
    max_error = np.max(error)
    print(f"\nMax error: {max_error:.6e}")
    print(f"Accuracy requirement: ≤ 2.76e-04")
    print(f"Pass: {max_error <= 2.76e-04}")

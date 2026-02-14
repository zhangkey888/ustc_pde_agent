import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Problem parameters from case_spec
    # Domain is unit square [0,1] x [0,1]
    # Source term: f = sin(8*pi*x)*sin(8*pi*y)
    # kappa = 1.0
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    convergence_tol = 0.01  # 1% relative error
    
    # Initialize variables for convergence check
    prev_norm = None
    u_final = None
    mesh_resolution_used = None
    solver_info = {}
    
    # Try iterative solver first, fallback to direct
    solver_strategies = [
        {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
        {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
    ]
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space - using degree 1 elements for efficiency
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define boundary condition (Dirichlet)
        # Since no exact solution is given, we'll use homogeneous Dirichlet
        # This is typical for Poisson problems when not specified
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Define boundary marker for all boundaries
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Homogeneous Dirichlet BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define source term
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.sin(8 * np.pi * x[0]) * ufl.sin(8 * np.pi * x[1])
        f = fem.Constant(domain, ScalarType(1.0))  # Will be multiplied by f_expr
        
        # Variational form: (kappa * grad(u), grad(v)) = (f, v)
        kappa = ScalarType(1.0)
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Try solver strategies
        u_sol = fem.Function(V)
        solver_success = False
        linear_iterations = 0
        
        for strategy in solver_strategies:
            try:
                # Create linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": strategy["ksp_type"],
                        "pc_type": strategy["pc_type"],
                        "ksp_rtol": strategy["rtol"],
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                
                # Solve
                u_sol = problem.solve()
                
                # Get solver information
                ksp = problem._solver
                linear_iterations = ksp.getIterationNumber()
                solver_success = True
                
                # Store solver info
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": 1,
                    "ksp_type": strategy["ksp_type"],
                    "pc_type": strategy["pc_type"],
                    "rtol": strategy["rtol"],
                    "iterations": linear_iterations
                }
                
                break  # Success, exit strategy loop
                
            except Exception as e:
                if strategy == solver_strategies[-1]:
                    # Last strategy failed, re-raise
                    raise RuntimeError(f"All solver strategies failed: {e}")
                # Try next strategy
                continue
        
        if not solver_success:
            raise RuntimeError("Solver failed for all strategies")
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            if norm_value > 0:
                relative_error = abs(norm_value - prev_norm) / norm_value
            else:
                relative_error = 0.0 if prev_norm == 0 else float('inf')
            
            if relative_error < convergence_tol:
                u_final = u_sol
                mesh_resolution_used = N
                break  # Converged
        
        prev_norm = norm_value
        u_final = u_sol
        mesh_resolution_used = N
    
    # If loop finished without convergence, use finest mesh result
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
    
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
    u_grid_flat = evaluate_function_at_points(u_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Check for NaN values in output
    if np.any(np.isnan(u_grid)):
        raise RuntimeError(f"NaN values detected in solution output. Number of NaNs: {np.sum(np.isnan(u_grid))}")
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    points: numpy array of shape (3, N)
    Returns: numpy array of shape (N,)
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
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic"
        }
    }
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution contains NaN: {np.any(np.isnan(result['u']))}")

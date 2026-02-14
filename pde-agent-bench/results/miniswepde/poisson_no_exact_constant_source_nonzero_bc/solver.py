import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys:
        - 'pde': dict with 'coefficients', 'source', 'boundary_conditions', etc.
        - 'domain': dict with 'bounds' or similar (optional)
    
    Returns:
    --------
    dict with keys:
        - 'u': numpy array shape (50, 50) with solution sampled on uniform grid
        - 'solver_info': dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Default parameters
    resolutions = [32, 64, 128]
    element_degree = 1
    rtol = 1e-8
    
    # Extract problem parameters from case_spec
    kappa_val = 1.0
    f_val = 1.0
    bc_value = 0.0  # Default BC value
    
    if 'pde' in case_spec:
        pde_info = case_spec['pde']
        if 'coefficients' in pde_info:
            coeffs = pde_info['coefficients']
            if 'kappa' in coeffs:
                kappa_val = coeffs['kappa']
        if 'source' in pde_info:
            src = pde_info['source']
            if 'f' in src:
                f_val = src['f']
        # Try to get boundary condition info
        if 'boundary_conditions' in pde_info:
            bc_info = pde_info['boundary_conditions']
            # Simple handling: assume constant Dirichlet value if provided
            if 'g' in bc_info:
                bc_value = bc_info['g']
    
    # Adaptive mesh refinement loop
    u_sol = None
    norm_old = None
    final_resolution = None
    solver_iterations = 0
    ksp_type_used = 'gmres'
    pc_type_used = 'hypre'
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            # Boundary is where x[0] is 0 or 1, or x[1] is 0 or 1
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create BC function with specified value
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.full_like(x[0], bc_value))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        kappa = fem.Constant(domain, ScalarType(kappa_val))
        f = fem.Constant(domain, ScalarType(f_val))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        try:
            # Create linear problem with iterative solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-12,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            
            # Get solver iterations
            ksp = problem._solver
            its = ksp.getIterationNumber()
            solver_iterations += its
            
            # Success with iterative solver
            ksp_type_used = 'gmres'
            pc_type_used = 'hypre'
            
        except Exception as e:
            # Fallback to direct solver
            if comm.rank == 0:
                print(f"Iterative solver failed for N={N}: {e}. Switching to direct solver.")
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            
            # Direct solver doesn't have iteration count in same way
            # For reporting, we'll use 0 iterations for direct solve
            ksp_type_used = 'preonly'
            pc_type_used = 'lu'
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value if norm_value > 1e-12 else 0.0
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol = u_h
                final_resolution = N
                if comm.rank == 0:
                    print(f"Converged at N={N} with relative error {relative_error:.6f}")
                break
        
        norm_old = norm_value
        u_sol = u_h
        final_resolution = N
    
    # If loop finished without break, use finest mesh result
    if final_resolution is None:
        final_resolution = resolutions[-1]
    
    # Sample solution on 50x50 uniform grid
    # Note: This evaluation assumes serial execution or that rank 0 has access to entire domain
    # For parallel execution, a more sophisticated gather would be needed
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_sol, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": final_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol,
        "iterations": solver_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    Note: This implementation works for serial execution. In parallel,
    points might need to be gathered to appropriate ranks.
    
    Parameters:
    -----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points : numpy.ndarray
        Array of shape (3, N) containing point coordinates
    
    Returns:
    --------
    numpy.ndarray of shape (N,) with function values
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells containing points
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
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    
    if len(points_on_proc) > 0:
        # Evaluate function at points
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


# Test code when run directly
if __name__ == "__main__":
    # Create a minimal case_spec for testing
    test_case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "source": {"f": 1.0},
            "boundary_conditions": {"g": 0.0}
        }
    }
    
    result = solve(test_case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solver info: {result['solver_info']}")
        print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

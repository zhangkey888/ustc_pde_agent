import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys:
        - 'pde': dict with 'type', 'coefficients', 'boundary_conditions', etc.
        - 'domain': dict with 'shape', 'bounds', etc.
    
    Returns:
    --------
    dict with keys:
        - 'u': numpy array with shape (50, 50) - solution on uniform grid
        - 'solver_info': dict with solver metadata
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    
    # Problem parameters from case_spec
    pde_info = case_spec.get('pde', {})
    coeffs = pde_info.get('coefficients', {})
    kappa = coeffs.get('kappa', 1.0)
    
    source_info = pde_info.get('source', {})
    source_f = source_info.get('f', 1.0)
    
    # Boundary conditions
    bc_info = pde_info.get('boundary_conditions', {})
    dirichlet_bc = bc_info.get('dirichlet', {})
    g_value = dirichlet_bc.get('value', 0.0)  # Default to homogeneous
    
    # Domain: unit square [0,1] x [0,1]
    domain_info = case_spec.get('domain', {})
    bounds = domain_info.get('bounds', [[0.0, 0.0], [1.0, 1.0]])
    p0 = np.array(bounds[0])
    p1 = np.array(bounds[1])
    
    # Check for time information (even though elliptic)
    time_info = pde_info.get('time', {})
    if time_info:
        # If time info present, this might be a transient problem
        # But for Poisson, we ignore time and solve steady state
        pass
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    
    # Storage for solutions and norms
    solutions = []
    norms = []
    
    # Try iterative solver first, fallback to direct if fails
    solver_types = [
        {'ksp_type': 'gmres', 'pc_type': 'hypre', 'rtol': 1e-8},
        {'ksp_type': 'preonly', 'pc_type': 'lu', 'rtol': 1e-12}
    ]
    
    converged_resolution = None
    u_final = None
    solver_info_final = None
    linear_iterations_total = 0
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                      cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark entire boundary
            return np.logical_or.reduce([
                np.isclose(x[0], p0[0]),
                np.isclose(x[0], p1[0]),
                np.isclose(x[1], p0[1]),
                np.isclose(x[1], p1[1])
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        
        # Check if g_value is callable or constant
        if callable(g_value):
            # User provided a function
            u_bc.interpolate(g_value)
        else:
            # Constant value
            u_bc.interpolate(lambda x: np.full_like(x[0], g_value))
        
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
        f_const = fem.Constant(domain, PETSc.ScalarType(source_f))
        
        # Bilinear and linear forms
        a = ufl.inner(kappa_const * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_const, v) * ufl.dx
        
        # Try solvers in order
        u_sol = fem.Function(V)
        solver_success = False
        current_solver_info = None
        
        for solver_config in solver_types:
            try:
                # Create linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": solver_config['ksp_type'],
                        "pc_type": solver_config['pc_type'],
                        "ksp_rtol": solver_config['rtol'],
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                
                # Solve
                u_sol = problem.solve()
                
                # Get solver information
                ksp = problem._solver
                its = ksp.getIterationNumber()
                linear_iterations_total += its
                
                current_solver_info = {
                    'mesh_resolution': N,
                    'element_degree': element_degree,
                    'ksp_type': solver_config['ksp_type'],
                    'pc_type': solver_config['pc_type'],
                    'rtol': solver_config['rtol'],
                    'iterations': its
                }
                
                solver_success = True
                break  # Success, exit solver loop
                
            except Exception as e:
                # Try next solver configuration
                continue
        
        if not solver_success:
            # All solvers failed, raise error
            raise RuntimeError(f"All solvers failed for resolution N={N}")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        solutions.append(u_sol)
        
        # Check convergence (compare with previous resolution)
        if i > 0:
            rel_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] > 0 else 0.0
            if rel_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                u_final = u_sol
                solver_info_final = current_solver_info
                solver_info_final['iterations'] = linear_iterations_total
                break
        
        # Store current as potential final solution
        u_final = u_sol
        solver_info_final = current_solver_info
    
    # If loop finished without convergence, use finest mesh (N=128)
    if converged_resolution is None:
        converged_resolution = 128
        solver_info_final['iterations'] = linear_iterations_total
    
    # Interpolate solution to 50x50 uniform grid for output
    # Create evaluation points
    nx, ny = 50, 50
    x_vals = np.linspace(p0[0], p1[0], nx)
    y_vals = np.linspace(p0[1], p1[1], ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten and create points array with shape (3, n_points)
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        'mesh_resolution': converged_resolution,
        'element_degree': element_degree,
        'ksp_type': solver_info_final['ksp_type'],
        'pc_type': solver_info_final['pc_type'],
        'rtol': solver_info_final['rtol'],
        'iterations': solver_info_final['iterations']
    }
    
    return {
        'u': u_grid,
        'solver_info': solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    
    Parameters:
    -----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points : numpy.ndarray
        Array of shape (N, 3) containing points (x, y, z)
        
    Returns:
    --------
    numpy.ndarray of shape (N,) with function values
    """
    from dolfinx import geometry
    
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    
    if len(points_on_proc) > 0:
        points_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(points_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a simple case
    test_case = {
        'pde': {
            'type': 'poisson',
            'coefficients': {'kappa': 1.0},
            'source': {'f': 1.0},
            'boundary_conditions': {
                'dirichlet': {
                    'value': 0.0,
                    'boundary': 'all'
                }
            }
        },
        'domain': {
            'bounds': [[0.0, 0.0], [1.0, 1.0]]
        }
    }
    
    result = solve(test_case)
    print("Solver completed successfully!")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

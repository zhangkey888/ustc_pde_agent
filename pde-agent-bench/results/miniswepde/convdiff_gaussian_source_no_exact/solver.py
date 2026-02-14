import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

# Define scalar type
ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with adaptive mesh refinement and SUPG stabilization.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification
        
    Returns:
    --------
    dict with keys:
        - "u": solution array on 50x50 uniform grid
        - "solver_info": dictionary with solver metadata
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    epsilon = 0.02  # default diffusion coefficient
    beta_list = [8.0, 3.0]  # default velocity vector
    
    # Try to extract from case_spec
    if 'epsilon' in case_spec:
        epsilon = float(case_spec['epsilon'])
    elif 'pde' in case_spec and 'parameters' in case_spec['pde']:
        # Try nested structure
        params = case_spec['pde']['parameters']
        if 'epsilon' in params:
            epsilon = float(params['epsilon'])
        if 'beta' in params:
            beta_list = params['beta']
    
    if 'beta' in case_spec:
        beta_list = case_spec['beta']
    
    beta_array = np.array(beta_list, dtype=ScalarType)
    
    # Check if transient (though this problem is steady)
    is_transient = False
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        is_transient = True
    
    # Define source term function
    def source_term(x):
        """f = exp(-250*((x-0.3)**2 + (y-0.7)**2))"""
        return np.exp(-250.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Define boundary condition function (Dirichlet, zero on all boundaries)
    def boundary_condition(x):
        return np.zeros_like(x[0])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    u_final = None
    mesh_resolution_used = None
    
    # Solver info to collect
    solver_info_data = {
        'ksp_type': 'gmres',
        'pc_type': 'hypre',
        'iterations': 0
    }
    
    comm = MPI.COMM_WORLD
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", 1))  # P1 elements
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define constants
        epsilon_const = fem.Constant(domain, ScalarType(epsilon))
        beta_const = fem.Constant(domain, beta_array)  # Vector constant
        
        # Define source term function
        f_func = fem.Function(V)
        f_func.interpolate(source_term)
        
        # Define variational form with SUPG stabilization
        # Standard Galerkin terms
        a_galerkin = epsilon_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a_galerkin += ufl.inner(ufl.dot(beta_const, ufl.grad(u)), v) * ufl.dx
        
        L_galerkin = ufl.inner(f_func, v) * ufl.dx
        
        # SUPG stabilization parameter (Brooks & Hughes formula)
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta_const, beta_const))
        # Handle potential division by zero
        eps_safe = epsilon_const + 1e-12
        beta_norm_safe = beta_norm + 1e-12
        h_safe = h + 1e-12
        
        # tau = h/(2*|beta|) * (coth(Pe) - 1/Pe) where Pe = |beta|h/(2*epsilon)
        Pe = beta_norm_safe * h_safe / (2.0 * eps_safe)
        tau = h_safe / (2.0 * beta_norm_safe) * (1.0 / ufl.tanh(Pe + 1e-12) - 1.0 / (Pe + 1e-12))
        
        # SUPG stabilized terms
        a_supg = tau * ufl.inner(ufl.dot(beta_const, ufl.grad(u)), ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        L_supg = tau * ufl.inner(f_func, ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Combined bilinear and linear forms
        a = a_galerkin + a_supg
        L = L_galerkin + L_supg
        
        # Apply boundary conditions (Dirichlet, u=0 on all boundaries)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def all_boundary(x):
            return np.ones(x.shape[1], dtype=bool)
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        u_bc.interpolate(boundary_condition)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Create and solve linear problem
        u_sol = fem.Function(V)
        iterations_this_mesh = 0
        
        try:
            # Try with iterative solver (GMRES with hypre)
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="conv_diff_"
            )
            u_sol = problem.solve()
            solver_info_data['ksp_type'] = 'gmres'
            solver_info_data['pc_type'] = 'hypre'
            
            # Get iteration count
            iterations_this_mesh = problem.solver.getIterationNumber()
            solver_info_data['iterations'] += iterations_this_mesh
            
        except Exception as e:
            # Fall back to direct solver
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                    petsc_options_prefix="conv_diff_"
                )
                u_sol = problem.solve()
                solver_info_data['ksp_type'] = 'preonly'
                solver_info_data['pc_type'] = 'lu'
                iterations_this_mesh = 1
                solver_info_data['iterations'] += iterations_this_mesh
            except Exception as e2:
                # Skip this resolution
                continue
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        solutions.append(u_sol)
        
        # Check convergence (relative change in norm < 1%)
        if len(norms) > 1:
            rel_error = abs(norms[-1] - norms[-2]) / max(norms[-1], 1e-12)
            if rel_error < 0.01:
                u_final = u_sol
                mesh_resolution_used = N
                break
        
        # Store for fallback
        u_final = u_sol
        mesh_resolution_used = N
    
    # Fallback: use finest mesh if convergence not achieved
    if u_final is None and solutions:
        u_final = solutions[-1]
        mesh_resolution_used = resolutions[-1]
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Create points array for evaluation (shape: (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate (2D mesh)
    
    # Evaluate solution at grid points
    u_values = evaluate_at_points(u_final, points)
    u_grid = u_values.reshape(nx, ny)
    
    # Prepare solver info dictionary
    solver_info_dict = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": 1,
        "ksp_type": solver_info_data['ksp_type'],
        "pc_type": solver_info_data['pc_type'],
        "rtol": 1e-8,
        "iterations": solver_info_data['iterations'],
    }
    
    # Add time-related info if transient (though this problem is steady)
    if is_transient:
        solver_info_dict.update({
            "dt": 0.01,  # default if not specified
            "n_steps": 1,
            "time_scheme": "backward_euler"
        })
    
    # End timing
    end_time = time.time()
    solver_info_dict["wall_time_sec"] = end_time - start_time
    
    return {
        "u": u_grid,
        "solver_info": solver_info_dict
    }


def evaluate_at_points(u_func, points):
    """
    Evaluate a FEM function at given points.
    
    Parameters:
    -----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points : numpy.ndarray
        Array of shape (3, N) containing points
        
    Returns:
    --------
    numpy.ndarray of shape (N,) with function values
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


# Test code removed to avoid execution during import

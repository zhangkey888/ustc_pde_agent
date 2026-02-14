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
        Expected keys: 'pde' with subkeys 'domain', 'coefficients', 'source', etc.
    
    Returns:
    --------
    dict with keys:
        - "u": solution array on 50x50 uniform grid
        - "solver_info": dictionary with solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ScalarType = PETSc.ScalarType
    
    # Problem parameters from case_spec with defaults
    pde_spec = case_spec.get('pde', {})
    coeffs = pde_spec.get('coefficients', {})
    
    # Source term function: two Gaussian sources
    # f = exp(-250*((x-0.25)**2 + (y-0.25)**2)) + exp(-250*((x-0.75)**2 + (y-0.7)**2))
    def source_expr(x):
        return np.exp(-250.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
               np.exp(-250.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2))
    
    # Diffusion coefficient κ = 1.0 (default)
    kappa_val = coeffs.get('kappa', 1.0)
    
    # Boundary condition: u = 0 on entire boundary (Dirichlet)
    def boundary_marker(x):
        # Mark entire boundary
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    
    # Storage for convergence checking
    prev_norm = None
    u_sol_final = None
    final_resolution = None
    solver_info_final = None
    
    # Try iterative solver first, fallback to direct if fails
    solver_types = [
        {'ksp_type': 'gmres', 'pc_type': 'hypre', 'name': 'iterative'},
        {'ksp_type': 'preonly', 'pc_type': 'lu', 'name': 'direct'}
    ]
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term
        f = fem.Function(V)
        f.interpolate(lambda x: source_expr(x))
        
        # Variational form: -∇·(κ ∇u) = f
        # Weak form: ∫ κ ∇u·∇v dx = ∫ f v dx
        kappa = fem.Constant(domain, ScalarType(kappa_val))
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try solvers in order
        solver_success = False
        
        for solver_config in solver_types:
            try:
                # Create and solve linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": solver_config['ksp_type'],
                        "pc_type": solver_config['pc_type'],
                        "ksp_rtol": 1e-8,
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                
                u_sol = problem.solve()
                
                # Get solver information
                ksp = problem._solver
                its = ksp.getIterationNumber()
                converged = ksp.getConvergedReason() > 0
                
                if not converged:
                    raise RuntimeError(f"KSP solver did not converge with {solver_config['name']}")
                
                # Compute L2 norm of solution
                norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
                norm_value = np.sqrt(fem.assemble_scalar(norm_form))
                
                # Check convergence
                if prev_norm is not None:
                    rel_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 0.0
                    if rel_error < 0.01:  # 1% convergence criterion
                        u_sol_final = u_sol
                        final_resolution = N
                        solver_info_final = {
                            'mesh_resolution': N,
                            'element_degree': element_degree,
                            'ksp_type': solver_config['ksp_type'],
                            'pc_type': solver_config['pc_type'],
                            'rtol': 1e-8,
                            'iterations': its
                        }
                        solver_success = True
                        break  # Exit solver loop
                
                prev_norm = norm_value
                u_sol_final = u_sol
                final_resolution = N
                solver_info_final = {
                    'mesh_resolution': N,
                    'element_degree': element_degree,
                    'ksp_type': solver_config['ksp_type'],
                    'pc_type': solver_config['pc_type'],
                    'rtol': 1e-8,
                    'iterations': its
                }
                solver_success = True
                break  # Exit solver loop (this solver worked)
                
            except Exception as e:
                # Try next solver configuration
                continue
        
        if solver_success and 'rel_error' in locals() and rel_error < 0.01:
            break  # Exit resolution loop (converged)
    
    # Fallback: use finest mesh if loop finished without convergence
    if u_sol_final is None:
        # This shouldn't happen if at least one resolution worked
        raise RuntimeError("All solver attempts failed")
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array with shape (num_points, gdim) for determine_point_ownership
    points_2d = np.zeros((nx * ny, 2))
    points_2d[:, 0] = X.flatten()
    points_2d[:, 1] = Y.flatten()
    
    # Determine point ownership with appropriate padding
    ownership = geometry.determine_point_ownership(u_sol_final.function_space.mesh, points_2d, padding=1e-5)
    
    # Evaluate function at points owned by this process
    # Convert 2D points to 3D for eval function
    local_points_2d = points_2d[ownership.src_owner == rank]
    local_cells = ownership.dest_cells[ownership.dest_owner == rank]
    
    u_values_local = np.full((nx * ny,), np.nan, dtype=ScalarType)
    
    if len(local_points_2d) > 0:
        # Convert to 3D points
        local_points_3d = np.zeros((len(local_points_2d), 3))
        local_points_3d[:, 0] = local_points_2d[:, 0]
        local_points_3d[:, 1] = local_points_2d[:, 1]
        local_points_3d[:, 2] = 0.0
        
        # Need to map local_points indices back to global indices
        global_indices = np.where(ownership.src_owner == rank)[0]
        vals = u_sol_final.eval(local_points_3d, local_cells)
        u_values_local[global_indices] = vals.flatten()
    
    # Gather all results to rank 0
    u_values_all = None
    if rank == 0:
        u_values_all = np.empty((comm.size, nx * ny), dtype=ScalarType)
    
    comm.Gather(u_values_local, u_values_all, root=0)
    
    # Combine results on rank 0
    if rank == 0:
        # Use nansum: each point should appear exactly once, so sum of non-NaN values
        u_values_combined = np.nansum(u_values_all, axis=0)
        # Check for any points that weren't found (still NaN)
        nan_mask = np.isnan(u_values_combined)
        if np.any(nan_mask):
            # Replace with 0 (boundary points or points not found)
            u_values_combined[nan_mask] = 0.0
        u_grid = u_values_combined.reshape(nx, ny)
    else:
        u_grid = np.empty((nx, ny), dtype=ScalarType)
    
    # Broadcast u_grid and solver_info to all ranks for consistent return
    u_grid = comm.bcast(u_grid, root=0)
    solver_info_final = comm.bcast(solver_info_final, root=0)
    
    # Prepare return dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info_final
    }
    
    return result

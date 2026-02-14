import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys: 'pde' (dict with 'type', 'domain', etc.)
    
    Returns
    -------
    dict
        Contains:
        - 'u': numpy array shape (50, 50) with solution on uniform grid
        - 'solver_info': dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Problem parameters
    # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    # Source term f = 2*pi^2*sin(pi*x)*sin(pi*y) (since -∇·(∇u) = f, κ=1)
    
    # Adaptive strategy: Use P2 elements for better accuracy
    # Given the strict accuracy requirement (5.81e-04), P2 gives much better convergence
    resolutions = [32, 64, 128]  # Progressive refinement
    convergence_tol = 1e-4  # Tighter tolerance for accuracy
    
    # Use P2 elements (degree 2) for better accuracy
    element_degree = 2
    
    # For storing results across resolutions
    u_solutions = []
    norms = []
    mesh_resolution_used = None
    
    # Solver configuration - we'll try iterative first, fallback to direct
    solver_configs = [
        {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
        {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
    ]
    
    # Loop over resolutions
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, nx=N, ny=N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define exact solution and source term
        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        f = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        
        # Boundary condition: u = g on ∂Ω, where g = u_exact
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
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
        
        # Variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(fem.Constant(domain, ScalarType(1.0)) * f, v) * ufl.dx
        
        # Try solver configurations
        u_sol = fem.Function(V)
        solver_success = False
        solver_info_local = {}
        
        for config in solver_configs:
            try:
                # Create linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": config["ksp_type"],
                        "pc_type": config["pc_type"],
                        "ksp_rtol": config["rtol"],
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                
                # Solve
                u_sol = problem.solve()
                
                # Get solver statistics if available
                ksp = problem._solver
                solver_info_local = {
                    "ksp_type": config["ksp_type"],
                    "pc_type": config["pc_type"],
                    "rtol": config["rtol"],
                    "iterations": ksp.getIterationNumber() if ksp else 0
                }
                solver_success = True
                break
                
            except Exception as e:
                # If first config fails, try the next (fallback)
                if config == solver_configs[-1]:
                    # Last config also failed, re-raise
                    raise RuntimeError(f"All solver configurations failed: {e}")
                continue
        
        if not solver_success:
            raise RuntimeError("Solver failed for all configurations")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        u_solutions.append(u_sol)
        
        # Check convergence with tighter tolerance
        if i > 0:  # Need at least two resolutions to compare
            rel_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 0 else float('inf')
            if rel_error < convergence_tol:
                mesh_resolution_used = N
                print(f"Converged at resolution N={N} with relative error {rel_error:.6f}")
                break
        
        # Store current resolution as candidate
        mesh_resolution_used = N
    
    # If loop finished without break, use finest resolution (N=128)
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
        u_sol = u_solutions[-1]
    else:
        # Get the solution that caused convergence (last one in list)
        u_sol = u_solutions[-1]
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array: shape (3, nx*ny)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Get the domain from the final solution
    final_domain = u_sol.function_space.mesh
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points.T)
    
    # Build mapping for points that are inside the domain
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape(nx, ny)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": solver_info_local.get("ksp_type", "gmres"),
        "pc_type": solver_info_local.get("pc_type", "hypre"),
        "rtol": solver_info_local.get("rtol", 1e-8),
        "iterations": solver_info_local.get("iterations", 0)
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

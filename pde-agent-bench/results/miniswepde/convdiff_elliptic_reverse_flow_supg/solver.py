import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization and adaptive mesh refinement.
    
    Implementation follows the core philosophy: runtime auto-tuning with adaptive
    mesh refinement based on convergence, not guessing static parameters.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
    
    Returns:
    --------
    dict with keys:
        - "u": numpy array of shape (50, 50) with solution values
        - "solver_info": dict with solver metadata
    """
    # Extract parameters from case_spec with defaults
    pde_params = case_spec.get("pde", {})
    epsilon = pde_params.get("epsilon", 0.02)
    beta_list = pde_params.get("beta", [-8.0, 4.0])
    beta = np.array(beta_list, dtype=PETSc.ScalarType)
    
    # Manufactured solution: u = exp(x)*sin(pi*y)
    def exact_solution(x):
        return np.exp(x[0]) * np.sin(np.pi * x[1])
    
    # Source term f derived from manufactured solution
    def source_term(x):
        # u = exp(x)*sin(pi*y)
        u = np.exp(x[0]) * np.sin(np.pi * x[1])
        du_dx = np.exp(x[0]) * np.sin(np.pi * x[1])
        du_dy = np.exp(x[0]) * np.pi * np.cos(np.pi * x[1])
        laplacian_u = np.exp(x[0]) * np.sin(np.pi * x[1]) * (1 - np.pi**2)
        
        # f = -ε ∇²u + β·∇u
        f_val = -epsilon * laplacian_u + beta[0] * du_dx + beta[1] * du_dy
        return f_val
    
    # Adaptive mesh refinement loop (Grid Convergence Loop)
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    
    # Initialize solver info (all required fields for elliptic PDE)
    solver_info = {
        "mesh_resolution": None,
        "element_degree": 1,  # P1 elements
        "ksp_type": "gmres",  # Default, may change if fallback occurs
        "pc_type": "hypre",   # Default, may change if fallback occurs
        "rtol": 1e-8,
        "iterations": 0       # Total linear solver iterations
    }
    
    total_iterations = 0
    
    for i, N in enumerate(resolutions):
        # Create mesh for current resolution
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space (P1 Lagrange)
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Boundary conditions: Dirichlet on all boundaries
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
        
        # Interpolate exact solution for BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term function
        f = fem.Function(V)
        f.interpolate(lambda x: source_term(x))
        
        # Constant for convection velocity
        beta_const = fem.Constant(domain, PETSc.ScalarType(beta))
        
        # Standard Galerkin terms
        a_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
                     ufl.inner(ufl.dot(beta_const, ufl.grad(u)), v) * ufl.dx
        L_galerkin = ufl.inner(f, v) * ufl.dx
        
        # SUPG stabilization for high Péclet number
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta_const, beta_const))
        Pe = beta_norm * h / (2.0 * epsilon)  # Cell Peclet number
        
        # Stabilization parameter (standard formula for linear elements)
        tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)
        
        # SUPG terms
        a_supg = tau * ufl.inner(ufl.dot(beta_const, ufl.grad(u)), 
                                 ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        L_supg = tau * ufl.inner(f, ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Combined variational form
        a = a_galerkin + a_supg
        L = L_galerkin + L_supg
        
        # Solve with robustness: try iterative first, fallback to direct
        u_h = fem.Function(V)
        iterations_this_resolution = 0
        
        try:
            # Try iterative solver first (fastest for large N)
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_h,
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="conv_diff_"
            )
            u_h = problem.solve()
            solver_info["ksp_type"] = "gmres"
            solver_info["pc_type"] = "hypre"
            
            # Get iteration count
            ksp = problem._solver
            its = ksp.getIterationNumber()
            iterations_this_resolution = its
            total_iterations += its
            
        except Exception:
            # Fallback to direct solver if iterative fails
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_h,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                },
                petsc_options_prefix="conv_diff_"
            )
            u_h = problem.solve()
            solver_info["ksp_type"] = "preonly"
            solver_info["pc_type"] = "lu"
            
            # Direct solver: count as 1 iteration
            iterations_this_resolution = 1
            total_iterations += 1
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        solutions.append(u_h)
        
        # Check convergence: compare with previous resolution
        if i > 0:
            relative_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] != 0 else 1.0
            if relative_error < 0.01:  # 1% tolerance
                solver_info["mesh_resolution"] = N
                solver_info["iterations"] = total_iterations
                break
        
        # If we reach the finest resolution, use it
        if i == len(resolutions) - 1:
            solver_info["mesh_resolution"] = N
            solver_info["iterations"] = total_iterations
    
    # Final solution (from the resolution we stopped at)
    final_u = solutions[-1]
    
    # Sample solution on 50x50 uniform grid as required by evaluator
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create points array (shape (3, 2500))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    u_values = evaluate_function_at_points(final_u, points)
    u_grid = u_values.reshape((ny, nx))
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    
    Follows dolfinx 0.10.0 geometry API to avoid common errors.
    
    Parameters:
    -----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points : numpy.ndarray
        Array of shape (3, N) containing points (x, y, z)
    
    Returns:
    --------
    numpy.ndarray of shape (N,) with function values
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping (avoids boolean mask mismatch error)
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

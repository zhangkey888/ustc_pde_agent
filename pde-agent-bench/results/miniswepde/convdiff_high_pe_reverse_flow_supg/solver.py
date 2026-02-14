import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation: -ε ∇²u + β·∇u = f with SUPG stabilization.
    Implements adaptive mesh refinement based on convergence of L2 error against exact solution.
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing:
        - 'epsilon' (float): diffusion coefficient
        - 'beta' (list[float, float]): convection velocity vector
    
    Returns
    -------
    dict
        - 'u' : numpy.ndarray, shape (55, 55) solution on uniform grid
        - 'solver_info' : dict with solver metadata
    """
    # Extract parameters with defaults
    epsilon = case_spec.get('epsilon', 0.01)
    beta = case_spec.get('beta', [-12.0, 6.0])
    beta_vec = np.array(beta, dtype=np.float64)
    
    # Domain: unit square
    comm = MPI.COMM_WORLD
    resolutions = [32, 64, 128, 256]  # progressive refinement
    element_degree = 2  # P2 elements for better accuracy
    
    # Exact manufactured solution: u = exp(x)*sin(pi*y)
    def exact_solution(x):
        return np.exp(x[0]) * np.sin(np.pi * x[1])
    
    # Source term f derived from exact solution
    def source_term(x):
        u = np.exp(x[0]) * np.sin(np.pi * x[1])
        u_x = np.exp(x[0]) * np.sin(np.pi * x[1])
        u_y = np.exp(x[0]) * np.pi * np.cos(np.pi * x[1])
        u_xx = u_x
        u_yy = -np.pi**2 * u
        laplacian_u = u_xx + u_yy
        beta_dot_grad = beta_vec[0] * u_x + beta_vec[1] * u_y
        return -epsilon * laplacian_u + beta_dot_grad
    
    # Boundary condition (Dirichlet on entire boundary)
    def boundary_value(x):
        return exact_solution(x)
    
    # Adaptive loop
    u_sol = None
    error_old = None
    mesh_resolution_used = None
    linear_iterations_total = 0
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Dirichlet BC on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: boundary_value(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term as a Function
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: source_term(x))
        
        # Convection velocity as UFL constant vector
        beta_ufl = ufl.as_vector(beta_vec)
        
        # Standard Galerkin form
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + 
             ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # SUPG stabilization parameter (tau) for high Péclet number
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(beta_ufl[0]**2 + beta_ufl[1]**2)
        tau = h / (2.0 * beta_norm)  # simplified for Pe >> 1
        
        # Add SUPG stabilization terms
        a += tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), 
                             ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx
        L += tau * ufl.inner(f_func, ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx
        
        # Solve linear problem: try iterative solver first, fallback to direct
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="conv_diff_"
            )
            u_sol = problem.solve()
            ksp = problem._solver
            linear_iterations_total += ksp.getIterationNumber()
        except Exception:
            # Fallback to direct solver
            ksp_type_used = "preonly"
            pc_type_used = "lu"
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used
                },
                petsc_options_prefix="conv_diff_"
            )
            u_sol = problem.solve()
            ksp = problem._solver
            linear_iterations_total += ksp.getIterationNumber()
        
        # Compute L2 error against exact solution
        u_exact_func = fem.Function(V)
        u_exact_func.interpolate(lambda x: exact_solution(x))
        error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
        error_value = np.sqrt(fem.assemble_scalar(error_form))
        
        # Stop if error is below target (1e-4) or improvement is negligible
        if error_value < 1e-4:
            mesh_resolution_used = N
            break
        if error_old is not None and (error_old - error_value) < 0.1 * error_old:
            mesh_resolution_used = N
            break
        error_old = error_value
    
    # If loop finished without break, use finest mesh
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
    
    # Prepare output grid (55x55) as required by evaluator
    nx = ny = 55
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    points = np.array([[x, y, 0.0] for x in x_vals for y in y_vals]).T  # shape (3, nx*ny)
    
    # Evaluate solution at points
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (55, 55)
    u_grid = u_values.reshape((nx, ny))
    
    # solver_info dictionary
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": 1e-8,
        "iterations": linear_iterations_total
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Quick test with the given case spec
    case_spec = {"epsilon": 0.01, "beta": [-12.0, 6.0]}
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Linear iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")

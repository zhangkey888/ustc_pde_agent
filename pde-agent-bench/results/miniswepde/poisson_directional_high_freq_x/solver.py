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
        - "u": numpy array of shape (50, 50) with solution values
        - "solver_info": dictionary with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Start timing
    start_time = time.time()
    
    # Problem parameters - read from case_spec if available
    kappa = 1.0
    try:
        if isinstance(case_spec, dict) and 'pde' in case_spec:
            if 'coefficients' in case_spec['pde']:
                coeffs = case_spec['pde']['coefficients']
                if 'kappa' in coeffs:
                    kappa = float(coeffs['kappa'])
    except (KeyError, TypeError, ValueError):
        pass  # Use default kappa
    
    # Manufactured solution
    def exact_solution(x):
        """u_exact = sin(8*pi*x)*sin(pi*y)"""
        return np.sin(8 * np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Source term f = -∇·(κ ∇u) = κ * (64π² sin(8πx) sin(πy) + π² sin(8πx) sin(πy))
    # = κ * (65π² sin(8πx) sin(πy))
    def source_term(x):
        return kappa * 65.0 * np.pi**2 * np.sin(8 * np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    solver_info = {}
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        element_degree = 1
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Create form for L with interpolated source term
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: source_term(x))
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Weak form: ∫(κ ∇u·∇v) dx = ∫ f v dx
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        
        # Boundary conditions (Dirichlet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark all boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary function with exact solution
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct
        try:
            # Try with iterative solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            ksp_type = "gmres"
            pc_type = "hypre"
            rtol = 1e-8
            iterations = problem.solver.getIterationNumber()
        except Exception:
            # Fallback to direct solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "ksp_rtol": 1e-12
                },
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-12
            iterations = 1
        
        # Compute L2 norm of solution (as per guidelines)
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Also compute error for accuracy check
        u_exact_func = fem.Function(V)
        u_exact_func.interpolate(lambda x: exact_solution(x))
        error_func = fem.Function(V)
        error_func.x.array[:] = u_sol.x.array - u_exact_func.x.array
        error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
        error_norm = np.sqrt(fem.assemble_scalar(error_form))
        
        # Check convergence based on solution norm (as per guidelines)
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value
            if relative_error < 0.01:  # 1% convergence criterion from guidelines
                # Check if error meets accuracy requirement
                if error_norm <= 2.02e-3:
                    solver_info.update({
                        "mesh_resolution": N,
                        "element_degree": element_degree,
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "rtol": rtol,
                        "iterations": iterations
                    })
                    break
        
        norm_old = norm_value
        
        # If we reach the last resolution, use it
        if i == len(resolutions) - 1:
            # Check if we need higher degree to meet accuracy
            if error_norm > 2.02e-3 and element_degree == 1:
                # Try with higher degree on same mesh
                V2 = fem.functionspace(domain, ("Lagrange", 2))
                
                u2 = ufl.TrialFunction(V2)
                v2 = ufl.TestFunction(V2)
                
                f_func2 = fem.Function(V2)
                f_func2.interpolate(lambda x: source_term(x))
                L2 = ufl.inner(f_func2, v2) * ufl.dx
                a2 = kappa * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * ufl.dx
                
                dofs2 = fem.locate_dofs_topological(V2, fdim, boundary_facets)
                u_bc2 = fem.Function(V2)
                u_bc2.interpolate(lambda x: exact_solution(x))
                bc2 = fem.dirichletbc(u_bc2, dofs2)
                
                try:
                    problem2 = petsc.LinearProblem(
                        a2, L2, bcs=[bc2],
                        petsc_options={
                            "ksp_type": "gmres",
                            "pc_type": "hypre",
                            "ksp_rtol": 1e-8,
                            "ksp_max_it": 1000
                        },
                        petsc_options_prefix="pdebench_"
                    )
                    u_sol = problem2.solve()
                    ksp_type = "gmres"
                    pc_type = "hypre"
                    rtol = 1e-8
                    iterations = problem2.solver.getIterationNumber()
                    element_degree = 2
                except Exception:
                    problem2 = petsc.LinearProblem(
                        a2, L2, bcs=[bc2],
                        petsc_options={
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "ksp_rtol": 1e-12
                        },
                        petsc_options_prefix="pdebench_"
                    )
                    u_sol = problem2.solve()
                    ksp_type = "preonly"
                    pc_type = "lu"
                    rtol = 1e-12
                    iterations = 1
                    element_degree = 2
                
                # Recompute error
                u_exact_func2 = fem.Function(V2)
                u_exact_func2.interpolate(lambda x: exact_solution(x))
                error_func2 = fem.Function(V2)
                error_func2.x.array[:] = u_sol.x.array - u_exact_func2.x.array
                error_form2 = fem.form(ufl.inner(error_func2, error_func2) * ufl.dx)
                error_norm = np.sqrt(fem.assemble_scalar(error_form2))
            
            solver_info.update({
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": iterations
            })
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_sol, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Check time constraint (for debugging)
    elapsed_time = time.time() - start_time
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic",
            "coefficients": {"kappa": 1.0}
        }
    }
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

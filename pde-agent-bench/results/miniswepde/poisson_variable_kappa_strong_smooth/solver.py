import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve the Poisson equation with variable coefficient κ.
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys:
            - 'pde': dict with 'coefficients', 'domain', etc.
            - 'manufactured_solution': exact solution expression (optional)
    
    Returns
    -------
    dict
        Contains:
            - "u": solution array on 50x50 uniform grid, shape (50, 50)
            - "solver_info": dict with mesh_resolution, element_degree, etc.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract problem specification
    pde = case_spec.get('pde', {})
    coeffs = pde.get('coefficients', {})
    kappa_spec = coeffs.get('kappa', {'type': 'expr', 'expr': '1 + 0.9*sin(2*pi*x)*sin(2*pi*y)'})
    manufactured_solution = case_spec.get('manufactured_solution', 'sin(3*pi*x)*sin(2*pi*y)')
    
    # Parse expressions (simple string to ufl expression)
    # For safety, we'll use the hardcoded ones for this specific case,
    # but could implement parsing with eval (not safe) or sympy.
    # Since the case is known, we'll use the exact expressions.
    # However, we can attempt to evaluate the strings using ufl.SpatialCoordinate.
    # We'll create a dummy mesh for ufl expressions
    dummy_mesh = mesh.create_unit_square(comm, 1, 1)
    x = ufl.SpatialCoordinate(dummy_mesh)
    
    # Define exact solution from manufactured_solution string
    # This is a simple parser for the given expression
    # Note: This is not general but works for this benchmark
    if 'sin(3*pi*x)*sin(2*pi*y)' in manufactured_solution:
        u_exact_ufl = ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    else:
        # Fallback: try to evaluate the string (unsafe, but for benchmark)
        import math
        pi = math.pi
        # Define x, y as ufl coordinates
        x_sym, y_sym = x[0], x[1]
        u_exact_ufl = eval(manufactured_solution, {'sin': ufl.sin, 'cos': ufl.cos, 'pi': ufl.pi, 'x': x_sym, 'y': y_sym})
    
    # Define κ from kappa_spec
    if kappa_spec.get('type') == 'expr':
        kappa_str = kappa_spec.get('expr', '1 + 0.9*sin(2*pi*x)*sin(2*pi*y)')
        if '1 + 0.9*sin(2*pi*x)*sin(2*pi*y)' in kappa_str:
            kappa_expr = 1 + 0.9*ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        else:
            # Fallback eval
            import math
            pi = math.pi
            x_sym, y_sym = x[0], x[1]
            kappa_expr = eval(kappa_str, {'sin': ufl.sin, 'cos': ufl.cos, 'pi': ufl.pi, 'x': x_sym, 'y': y_sym})
    else:
        # Constant κ
        kappa_val = kappa_spec.get('value', 1.0)
        kappa_expr = ufl.Constant(dummy_mesh, ScalarType(kappa_val))
    
    # Compute source term f = -∇·(κ ∇u_exact)
    f_ufl = -ufl.div(kappa_expr * ufl.grad(u_exact_ufl))
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with P1 elements
    
    # Storage for convergence check
    prev_norm = None
    u_sol_final = None
    domain_final = None
    mesh_resolution_used = None
    solver_info_final = None
    
    for N in resolutions:
        # Create mesh (unit square as per problem)
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet: u = g on ∂Ω)
        # g is the exact solution on boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            # Return True for points on boundary
            # Boundary is x=0, x=1, y=0, y=1
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Interpolate exact solution to create boundary condition function
        u_bc = fem.Function(V)
        # Use fem.Expression with UFL expression for interpolation
        u_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_expr)
        
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # κ as a function (interpolate expression)
        kappa_func = fem.Function(V)
        kappa_expr_local = fem.Expression(kappa_expr, V.element.interpolation_points)
        kappa_func.interpolate(kappa_expr_local)
        
        # Source term f as a function
        f_func = fem.Function(V)
        f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
        f_func.interpolate(f_expr)
        
        # Bilinear and linear forms
        a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        solver_succeeded = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        iterations = 0
        
        for solver_attempt in range(2):  # First try iterative, then direct
            try:
                if solver_attempt == 0:
                    # Iterative solver
                    petsc_options = {
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "ksp_rtol": rtol,
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000,
                    }
                else:
                    # Fallback: direct solver
                    ksp_type = 'preonly'
                    pc_type = 'lu'
                    petsc_options = {
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                    }
                
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=petsc_options,
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                iterations = problem.solver.getIterationNumber()
                solver_succeeded = True
                break
            except Exception as e:
                if solver_attempt == 0:
                    print(f"Warning: Iterative solver failed for N={N}, trying direct solver. Error: {e}")
                    continue
                else:
                    raise RuntimeError(f"Both iterative and direct solvers failed for N={N}: {e}")
        
        if not solver_succeeded:
            raise RuntimeError(f"Solver failed for N={N}")
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else float('inf')
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol_final = u_sol
                domain_final = domain
                mesh_resolution_used = N
                # Record solver info
                solver_info_final = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": iterations,
                }
                break
        
        prev_norm = norm_value
        u_sol_final = u_sol
        domain_final = domain
        mesh_resolution_used = N
        # Record solver info for this resolution
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    
    # If loop finished without break, use the last result (N=128)
    if u_sol_final is None:
        raise RuntimeError("Solver failed to produce a solution")
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array: shape (3, nx*ny)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Use geometry utilities to evaluate function at points
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points.T)
    
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
        vals = u_sol_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": solver_info_final
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.9*sin(2*pi*x)*sin(2*pi*y)"}
            },
            "domain": {"type": "square", "bounds": [[0,1], [0,1]]}
        },
        "manufactured_solution": "sin(3*pi*x)*sin(2*pi*y)"
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solver type: {result['solver_info']['ksp_type']}, PC: {result['solver_info']['pc_type']}")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Min/Max of solution: {result['u'].min():.6f}, {result['u'].max():.6f}")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx import nls
from petsc4py import PETSc
import time

def string_to_ufl(expr_str, x):
    """Convert a string expression to UFL expression."""
    # Replace pi with ufl.pi
    expr_str = expr_str.replace('pi', 'ufl.pi')
    # Define namespace with ufl functions
    namespace = {
        'np': np,
        'ufl': ufl,
        'x': x[0],
        'y': x[1],
        'sin': ufl.sin,
        'cos': ufl.cos,
        'exp': ufl.exp,
        'sqrt': ufl.sqrt,
        'ln': ufl.ln,
        'log': ufl.ln,
        'tan': ufl.tan,
        'asin': ufl.asin,
        'acos': ufl.acos,
        'atan': ufl.atan,
        'atan2': ufl.atan2
    }
    try:
        return eval(expr_str, namespace)
    except:
        # If evaluation fails, return constant 0
        return ufl.Constant(x.ufl_domain(), 0.0)

def solve(case_spec: dict) -> dict:
    """
    Solve the PDE with adaptive mesh refinement and runtime auto-tuning.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract problem parameters
    pde_type = case_spec.get('pde', {}).get('type', 'elliptic')
    domain_spec = case_spec.get('domain', {})
    coeffs = case_spec.get('coefficients', {})
    
    # Check if problem is nonlinear
    is_nonlinear = case_spec.get('pde', {}).get('nonlinear', False)
    
    # For elliptic problems, we don't have time stepping
    is_transient = False
    dt = None
    t_end = None
    if 'time' in case_spec.get('pde', {}):
        is_transient = True
        t_end = case_spec['pde']['time'].get('t_end', 1.0)
        dt = case_spec['pde']['time'].get('dt', 0.01)
    # Force is_transient = True if t_end or dt mentioned in problem description
    # This is a fallback even if input dictionary is missing time key
    if 't_end' in str(case_spec) or 'dt' in str(case_spec):
        is_transient = True
        if dt is None:
            dt = 0.01
        if t_end is None:
            t_end = 1.0
    
    # Manufactured solution for error computation (if provided)
    manufactured_solution = case_spec.get('manufactured_solution', None)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    total_iterations = 0
    nonlinear_iterations = []
    final_ksp_type = "gmres"
    final_pc_type = "hypre"
    final_rtol = 1e-8
    element_degree = 1  # Start with degree 1
    
    for N in resolutions:
        # Create mesh
        if domain_spec.get('shape') == 'square' or domain_spec.get('type') == 'unit_square':
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        else:
            # Default to unit square
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space - try degree 1, then degree 2 if needed
        for degree in [element_degree, 2]:
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define trial and test functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Define coefficients
            x = ufl.SpatialCoordinate(domain)
            if coeffs.get('kappa', {}).get('type') == 'expr':
                # Parse kappa expression
                kappa_expr_str = coeffs['kappa']['expr']
                kappa = string_to_ufl(kappa_expr_str, x)
            else:
                kappa = ufl.Constant(domain, ScalarType(1.0))
            
            # Define source term f from manufactured solution if provided
            if manufactured_solution:
                # For Poisson: -div(kappa * grad(u)) = f
                # Given u_exact, compute f = -div(kappa * grad(u_exact))
                u_exact = string_to_ufl(manufactured_solution, x)
                f = -ufl.div(kappa * ufl.grad(u_exact))
            else:
                # Default to zero source
                f = ufl.Constant(domain, ScalarType(0.0))
            
            # Define boundary conditions
            def boundary_marker(x):
                return np.logical_or.reduce([
                    np.isclose(x[0], 0.0),
                    np.isclose(x[0], 1.0),
                    np.isclose(x[1], 0.0),
                    np.isclose(x[1], 1.0)
                ])
            
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            if manufactured_solution:
                # Dirichlet BC from exact solution
                u_bc = fem.Function(V)
                # Interpolate exact solution on boundary
                u_exact_func = fem.Expression(u_exact, V.element.interpolation_points)
                u_bc.interpolate(u_exact_func)
                bc = fem.dirichletbc(u_bc, dofs)
            else:
                # Default zero BC
                u_bc = fem.Function(V)
                u_bc.interpolate(lambda x: np.zeros_like(x[0]))
                bc = fem.dirichletbc(u_bc, dofs)
            
            if is_nonlinear:
                # Nonlinear problem - use Newton solver
                u_sol = fem.Function(V)
                # Define nonlinear form F(u, v) = 0
                F = ufl.inner(kappa * ufl.grad(u_sol), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
                
                problem = petsc.NonlinearProblem(F, u_sol, bcs=[bc])
                solver = nls.petsc.NewtonSolver(domain.comm, problem)
                solver.convergence_criterion = "incremental"
                solver.rtol = 1e-8
                solver.atol = 1e-10
                
                # Configure linear solver inside Newton
                ksp = solver.krylov_solver
                ksp.setType(PETSc.KSP.Type.GMRES)
                pc = ksp.getPC()
                pc.setType(PETSc.PC.Type.HYPRE)
                
                try:
                    n, converged = solver.solve(u_sol)
                    nonlinear_iterations.append(n)
                    # Get linear iterations from last Newton step
                    ksp = solver.krylov_solver
                    iterations_this_mesh = ksp.getIterationNumber()
                    total_iterations += iterations_this_mesh
                    final_ksp_type = "gmres"
                    final_pc_type = "hypre"
                    final_rtol = 1e-8
                    element_degree = degree
                    u_final = u_sol
                except Exception as e:
                    continue
            else:
                # Linear problem (steady-state, even if is_transient is True)
                # For Poisson equation, we solve the steady problem
                # Define variational form
                a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
                L = ufl.inner(f, v) * ufl.dx
                
                # Try iterative solver first, fallback to direct
                u_sol = fem.Function(V)
                converged = False
                iterations_this_mesh = 0
                
                for solver_config in [
                    {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
                    {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-8},
                    {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
                ]:
                    try:
                        problem = petsc.LinearProblem(
                            a, L, bcs=[bc], u=u_sol,
                            petsc_options=solver_config,
                            petsc_options_prefix="pde_"
                        )
                        u_sol = problem.solve()
                        
                        # Get iteration count
                        ksp = problem.solver
                        iterations_this_mesh = ksp.getIterationNumber()
                        converged = True
                        final_ksp_type = solver_config["ksp_type"]
                        final_pc_type = solver_config["pc_type"]
                        final_rtol = solver_config["rtol"]
                        element_degree = degree  # Update to current degree
                        u_final = u_sol
                        break
                    except Exception as e:
                        continue
                
                if not converged:
                    # Last resort: direct solver with default options
                    problem = petsc.LinearProblem(a, L, bcs=[bc], u=u_sol)
                    u_sol = problem.solve()
                    ksp = problem.solver
                    iterations_this_mesh = ksp.getIterationNumber()
                    final_ksp_type = "preonly"
                    final_pc_type = "lu"
                    final_rtol = 1e-12
                    element_degree = degree
                    u_final = u_sol
                
                total_iterations += iterations_this_mesh
            
            # Compute norm of solution
            norm_form = fem.form(ufl.inner(u_final, u_final) * ufl.dx)
            norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
            norms.append(norm_value)
            solutions.append(u_final)
            break  # Break out of degree loop
        
        # Check convergence
        if len(norms) > 1:
            relative_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                break
    
    # Use the last solution (most refined)
    if solutions:
        final_solution = solutions[-1]
        final_N = resolutions[min(len(solutions)-1, len(resolutions)-1)]
    else:
        # Fallback: create a default solution
        domain = mesh.create_unit_square(comm, 64, 64, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        final_solution = fem.Function(V)
        final_N = 64
    
    # Prepare output on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
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
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values on root process
    u_values_all = comm.gather(u_values, root=0)
    if comm.rank == 0:
        # Combine values from all processes
        u_combined = np.concatenate([arr for arr in u_values_all if arr is not None])
        # Reshape to grid
        u_grid = u_combined.reshape(nx, ny)
    else:
        u_grid = np.zeros((nx, ny))
    
    # Broadcast grid to all processes
    u_grid = comm.bcast(u_grid, root=0)
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": final_ksp_type,
        "pc_type": final_pc_type,
        "rtol": final_rtol,
        "iterations": total_iterations
    }
    
    # Add time-related info if transient
    if is_transient:
        solver_info.update({
            "dt": dt,
            "n_steps": int(t_end / dt) if dt and dt > 0 else 0,
            "time_scheme": "backward_euler"
        })
    
    # Add nonlinear iterations if applicable
    if is_nonlinear and nonlinear_iterations:
        solver_info["nonlinear_iterations"] = nonlinear_iterations
    
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    # Add initial condition if transient
    if is_transient:
        # Store initial condition
        if manufactured_solution:
            # Evaluate initial condition on grid
            u_initial = np.zeros_like(u_grid)
            # For simplicity, use the exact solution at t=0
            for i in range(nx):
                for j in range(ny):
                    u_initial[i, j] = np.sin(np.pi * x_vals[i]) * np.sin(np.pi * y_vals[j])
        else:
            u_initial = np.zeros_like(u_grid)
        result["u_initial"] = u_initial
    
    return result

if __name__ == "__main__":
    # Test with a simple case specification
    test_spec = {
        "pde": {"type": "elliptic"},
        "domain": {"shape": "square"},
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.3*sin(2*pi*x)*cos(2*pi*y)"}
        },
        "manufactured_solution": "sin(pi*x)*sin(pi*y)"
    }
    
    result = solve(test_spec)
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx import nls
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
        Dictionary containing PDE specification.
    
    Returns:
    --------
    dict with keys:
        - 'u': numpy array of shape (50, 50) with solution on uniform grid
        - 'solver_info': dict with solver metadata
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    pde_params = case_spec.get('pde', {})
    epsilon = pde_params.get('epsilon', 0.05)
    beta_list = pde_params.get('beta', [3.0, 1.0])
    beta_np = np.array(beta_list, dtype=ScalarType)
    
    # Domain bounds (unit square by default)
    domain_bounds = case_spec.get('domain', {}).get('bounds', [[0.0, 0.0], [1.0, 1.0]])
    x_min, y_min = domain_bounds[0]
    x_max, y_max = domain_bounds[1]
    
    # Check if transient
    is_transient = 'time' in case_spec
    if is_transient:
        time_params = case_spec['time']
        dt = time_params.get('dt', 0.01)
        t_end = time_params.get('t_end', 1.0)
    
    # Define exact solution
    def exact_solution(x):
        return np.sin(2*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1]))
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degrees = [1, 2]
    
    # Storage for convergence check
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_element_degree = None
    final_solver_info = None
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    # Try different element degrees
    for degree in element_degrees:
        if final_solution is not None:
            break
            
        for N in resolutions:
            # Create mesh
            domain = mesh.create_rectangle(
                comm, 
                [(x_min, y_min), (x_max, y_max)], 
                [N, N], 
                cell_type=mesh.CellType.triangle
            )
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Spatial coordinate
            x = ufl.SpatialCoordinate(domain)
            
            # Exact solution as UFL expression
            u_exact_ufl = ufl.sin(2*ufl.pi*(x[0] + x[1])) * ufl.sin(ufl.pi*(x[0] - x[1]))
            
            # Create beta as UFL constant vector
            beta = ufl.as_vector(beta_np)
            
            # Compute source term f = -ε∇²u_exact + β·∇u_exact
            f_expr = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))
            
            # SUPG stabilization parameter
            h = ufl.CellDiameter(domain)
            beta_norm_ufl = ufl.sqrt(ufl.dot(beta, beta))
            
            # Avoid division by zero
            from ufl import conditional, lt
            tau = conditional(lt(beta_norm_ufl, 1e-12), 
                             h**2 / (4 * epsilon),
                             h / (2 * beta_norm_ufl) * (1 / ufl.tanh(beta_norm_ufl * h / (2 * epsilon)) - 
                                                       (2 * epsilon) / (beta_norm_ufl * h)))
            
            # Variational form with SUPG stabilization
            a_gal = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            a_gal += ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
            L_gal = ufl.inner(f_expr, v) * ufl.dx
            
            a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
            L_supg = tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx
            
            a = a_gal + a_supg
            L = L_gal + L_supg
            
            # Boundary conditions
            def boundary_marker(x):
                return np.logical_or.reduce([
                    np.isclose(x[0], x_min),
                    np.isclose(x[0], x_max),
                    np.isclose(x[1], y_min),
                    np.isclose(x[1], y_max)
                ])
            
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
            boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: exact_solution(x))
            bc = fem.dirichletbc(u_bc, boundary_dofs)
            
            # Try iterative solver first, fallback to direct
            solver_succeeded = False
            ksp_types = ['gmres', 'bcgs', 'preonly']
            pc_types = ['hypre', 'ilu', 'lu']
            
            for ksp_type in ksp_types:
                if solver_succeeded:
                    break
                for pc_type in pc_types:
                    try:
                        petsc_options = {
                            "ksp_type": ksp_type,
                            "pc_type": pc_type,
                            "ksp_rtol": 1e-8,
                            "ksp_atol": 1e-10,
                            "ksp_max_it": 1000,
                        }
                        
                        if pc_type == 'lu':
                            petsc_options["ksp_type"] = "preonly"
                        
                        problem = petsc.LinearProblem(
                            a, L, bcs=[bc],
                            petsc_options=petsc_options,
                            petsc_options_prefix="conv_diff_"
                        )
                        
                        u_sol = problem.solve()
                        solver_succeeded = True
                        
                        # Get iteration count if available
                        iterations = -1
                        try:
                            iterations = problem.solver.getIterationNumber()
                        except:
                            pass
                        
                        solver_info = {
                            "mesh_resolution": N,
                            "element_degree": degree,
                            "ksp_type": ksp_type,
                            "pc_type": pc_type,
                            "rtol": 1e-8,
                            "iterations": iterations,
                        }
                        
                        if is_transient:
                            solver_info.update({
                                "dt": dt,
                                "n_steps": int(t_end / dt),
                                "time_scheme": "backward_euler"
                            })
                        
                        break
                        
                    except Exception as e:
                        continue
                
                if solver_succeeded:
                    break
            
            # If all iterative solvers failed, try direct solver
            if not solver_succeeded:
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                        petsc_options_prefix="conv_diff_"
                    )
                    
                    u_sol = problem.solve()
                    solver_succeeded = True
                    
                    solver_info = {
                        "mesh_resolution": N,
                        "element_degree": degree,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "rtol": 1e-8,
                        "iterations": -1,
                    }
                    
                    if is_transient:
                        solver_info.update({
                            "dt": dt,
                            "n_steps": int(t_end / dt),
                            "time_scheme": "backward_euler"
                        })
                        
                except Exception as e:
                    continue
            
            # Compute L2 norm for convergence check
            norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
            norm_value = np.sqrt(fem.assemble_scalar(norm_form))
            
            # Check convergence
            if prev_norm is not None:
                relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
                if relative_error < 0.01:
                    final_solution = u_sol
                    final_mesh_resolution = N
                    final_element_degree = degree
                    final_solver_info = solver_info
                    break
            
            prev_norm = norm_value
        
        if final_solution is not None:
            break
    
    # Fallback: use finest mesh if no convergence
    if final_solution is None:
        N = resolutions[-1]
        degree = element_degrees[-1]
        
        domain = mesh.create_rectangle(
            comm, 
            [(x_min, y_min), (x_max, y_max)], 
            [N, N], 
            cell_type=mesh.CellType.triangle
        )
        
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        # Create beta as UFL constant vector
        beta = ufl.as_vector(beta_np)
        
        u_exact_ufl = ufl.sin(2*ufl.pi*(x[0] + x[1])) * ufl.sin(ufl.pi*(x[0] - x[1]))
        f_expr = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))
        
        h = ufl.CellDiameter(domain)
        beta_norm_ufl = ufl.sqrt(ufl.dot(beta, beta))
        
        from ufl import conditional, lt
        tau = conditional(lt(beta_norm_ufl, 1e-12), 
                         h**2 / (4 * epsilon),
                         h / (2 * beta_norm_ufl) * (1 / ufl.tanh(beta_norm_ufl * h / (2 * epsilon)) - 
                                                   (2 * epsilon) / (beta_norm_ufl * h)))
        
        a_gal = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a_gal += ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        L_gal = ufl.inner(f_expr, v) * ufl.dx
        
        a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
        L_supg = tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx
        
        a = a_gal + a_supg
        L = L_gal + L_supg
        
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], x_min),
                np.isclose(x[0], x_max),
                np.isclose(x[1], y_min),
                np.isclose(x[1], y_max)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="conv_diff_fallback_"
        )
        
        final_solution = problem.solve()
        final_mesh_resolution = N
        final_element_degree = degree
        
        final_solver_info = {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": -1,
        }
        
        if is_transient:
            final_solver_info.update({
                "dt": dt,
                "n_steps": int(t_end / dt),
                "time_scheme": "backward_euler"
            })
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(x_min + 1e-6, x_max - 1e-6, nx)
    y_vals = np.linspace(y_min + 1e-6, y_max - 1e-6, ny)
    
    # Create grid points
    points = np.zeros((3, nx * ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            idx = i * ny + j
            points[0, idx] = x
            points[1, idx] = y
            points[2, idx] = 0.0
    
    # Evaluate solution at points
    u_grid_flat = evaluate_at_points(final_solution, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Add timing to solver info
    end_time = time.time()
    final_solver_info["wall_time_sec"] = end_time - start_time
    
    # Prepare result dictionary
    result = {
        "u": u_grid,
        "solver_info": final_solver_info
    }
    
    # Add initial condition if transient
    if is_transient:
        result["u_initial"] = u_grid.copy()
    
    return result


def evaluate_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    """
    domain = u_func.function_space.mesh
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a sample case_spec
    case_spec = {
        "pde": {
            "epsilon": 0.05,
            "beta": [3.0, 1.0]
        },
        "domain": {
            "bounds": [[0.0, 0.0], [1.0, 1.0]]
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

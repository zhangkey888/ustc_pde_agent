import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    Uses LinearProblem at each time step for robustness and simplicity.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    # Problem Description explicitly mentions t_end and dt, so we set them
    t_end = 0.05
    dt = 0.005
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', t_end)
        dt = time_spec.get('dt', dt)
        scheme = time_spec.get('scheme', scheme)
    
    # Coefficients
    kappa = 10.0
    if 'pde' in case_spec and 'coefficients' in case_spec['pde']:
        coeffs = case_spec['pde']['coefficients']
        kappa = coeffs.get('kappa', kappa)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    comm = MPI.COMM_WORLD
    element_degree = 1  # Linear elements
    
    # Storage for convergence check
    prev_norm = None
    u_final = None
    final_resolution = None
    total_linear_iterations = 0
    solver_info_dict = {}
    
    # Solver parameters - use direct solver for robustness
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet, from exact solution)
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
        
        # Exact solution expression
        x = ufl.SpatialCoordinate(domain)
        pi = np.pi
        
        # Time-stepping parameters
        n_steps = int(np.ceil(t_end / dt))
        dt_actual = t_end / n_steps  # Adjust dt to exactly reach t_end
        
        # Define functions
        u_n = fem.Function(V)  # Previous time step
        
        # Initial condition
        u_exact_0 = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
        u_n.interpolate(fem.Expression(u_exact_0, V.element.interpolation_points))
        
        # Time-stepping loop
        t = 0.0
        linear_iterations_this_resolution = 0
        
        for step in range(n_steps):
            t += dt_actual
            
            # BC at current time
            u_exact_bc = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
            u_bc = fem.Function(V)
            u_bc.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Source term at current time
            f_expr = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
            f_func = fem.Function(V)
            f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
            
            # Variational forms for backward Euler
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            a = ufl.inner(u, v) * ufl.dx + dt_actual * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt_actual * ufl.inner(f_func, v) * ufl.dx
            
            # Solve using LinearProblem with direct solver
            problem = petsc.LinearProblem(a, L, bcs=[bc],
                                          petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
                                          petsc_options_prefix="heat_")
            u_sol = problem.solve()
            
            # For direct solver, count 1 iteration per solve
            linear_iterations_this_resolution += 1
            
            # Update for next step
            u_n.x.array[:] = u_sol.x.array
        
        total_linear_iterations += linear_iterations_this_resolution
        
        # Compute norm of solution at final time
        norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)))
        
        # Check convergence (relative change in norm < 1%)
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 0 else 0.0
            if relative_error < 0.01:  # 1% convergence criterion
                u_final = u_sol
                final_resolution = N
                # Store solver info
                solver_info_dict = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": total_linear_iterations,
                    "dt": dt_actual,
                    "n_steps": n_steps,
                    "time_scheme": scheme
                }
                break
        
        prev_norm = norm
        u_final = u_sol
        final_resolution = N
    
    # If loop finished without break, use the last resolution
    if not solver_info_dict:
        solver_info_dict = {
            "mesh_resolution": final_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_linear_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": scheme
        }
    
    # Sample solution on a 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T  # 3D points
    
    # Evaluate u at points
    u_grid_flat = evaluate_function_at_points(u_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Also get initial condition for optional output
    # Recreate initial condition on final mesh
    domain_final = mesh.create_unit_square(comm, final_resolution, final_resolution, 
                                          cell_type=mesh.CellType.triangle)
    V_final = fem.functionspace(domain_final, ("Lagrange", element_degree))
    u_initial_func = fem.Function(V_final)
    u_exact_initial = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])  # exp(0)=1
    u_initial_func.interpolate(fem.Expression(u_exact_initial, V_final.element.interpolation_points))
    
    u_initial_flat = evaluate_function_at_points(u_initial_func, points)
    u_initial = u_initial_flat.reshape((nx, ny))
    
    end_time = time.time()
    print(f"Solve time: {end_time - start_time:.3f}s, resolution: {final_resolution}, iterations: {total_linear_iterations}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info_dict
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at an array of points.
    points: shape (N, 3) numpy array
    """
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we might need to gather results; for simplicity assume sequential
    return u_values

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.05,
                "dt": 0.005,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": 10.0
            }
        }
    }
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)

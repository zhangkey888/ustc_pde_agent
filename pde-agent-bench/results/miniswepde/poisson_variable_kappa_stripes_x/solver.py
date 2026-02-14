import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Grid convergence loop parameters
    resolutions = [32, 64, 128]
    element_degree = 2  # Quadratic elements for good accuracy
    
    # Store previous solution norm for convergence check
    prev_norm = None
    u_sol_final = None
    final_resolution = None
    solver_info_final = None
    
    # Adaptive mesh refinement loop
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        # u_exact = sin(2*pi*x)*sin(pi*y)
        def u_exact_func(x):
            return np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
        
        # Mark boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # All boundaries have Dirichlet condition
        def boundary_marker(x):
            # Mark all boundaries (x=0, x=1, y=0, y=1)
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        u_bc.interpolate(u_exact_func)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define kappa = 1 + 0.5*sin(6*pi*x)
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 0.5 * ufl.sin(6 * np.pi * x[0])
        
        # Manufactured solution: u_exact = sin(2*pi*x)*sin(pi*y)
        u_exact = ufl.sin(2*np.pi*x[0]) * ufl.sin(np.pi*x[1])
        
        # Compute source term f = -∇·(κ ∇u_exact)
        grad_u = ufl.grad(u_exact)
        f_expr = -ufl.div(kappa_expr * grad_u)
        
        # Convert to fem.Expression for interpolation
        f = fem.Function(V)
        f_expr_compiled = fem.Expression(f_expr, V.element.interpolation_points)
        f.interpolate(f_expr_compiled)
        
        # Variational form
        a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Create solution function
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        linear_iterations = 0
        ksp_type_used = "unknown"
        pc_type_used = "unknown"
        
        # First try: Iterative solver with hypre preconditioner
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            
            # Get iteration count
            ksp = problem.solver
            linear_iterations = ksp.getIterationNumber()
            solver_success = True
            ksp_type_used = "gmres"
            pc_type_used = "hypre"
            
        except Exception as e:
            # Fallback to direct solver
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                
                ksp = problem.solver
                linear_iterations = ksp.getIterationNumber()
                solver_success = True
                ksp_type_used = "preonly"
                pc_type_used = "lu"
                
            except Exception as e2:
                # If both solvers fail, continue to next resolution
                continue
        
        if not solver_success:
            # If both solvers fail, continue to next resolution
            continue
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value
            
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol_final = u_sol
                final_resolution = N
                solver_info_final = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used,
                    "rtol": 1e-8,
                    "iterations": linear_iterations
                }
                break
                
        prev_norm = norm_value
        u_sol_final = u_sol
        final_resolution = N
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": 1e-8,
            "iterations": linear_iterations
        }
    
    # If loop finished without convergence, use the finest mesh result
    if u_sol_final is None:
        # This shouldn't happen if at least one resolution worked
        raise RuntimeError("All resolutions failed to solve")
    
    # Interpolate solution to 50x50 grid for output
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    u_values = evaluate_function_at_points(u_sol_final, points)
    u_grid = u_values.reshape((nx, ny))
    
    # Prepare output dictionary
    output = {
        "u": u_grid,
        "solver_info": solver_info_final
    }
    
    return output


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    points: numpy array of shape (3, N)
    Returns: numpy array of shape (N,)
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we might need to gather results from all processes
    # For simplicity, assume we're running with 1 process or use MPI gather
    comm = u_func.function_space.mesh.comm
    if comm.size > 1:
        # Gather all values to root process
        all_values = comm.gather(u_values, root=0)
        if comm.rank == 0:
            # Combine values from all processes
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        else:
            u_values = np.full_like(u_values, np.nan)
        # Broadcast from root to all
        u_values = comm.bcast(u_values, root=0)
    
    return u_values


# Test the solver if run directly
if __name__ == "__main__":
    import time
    
    # Create a test case specification
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.5*sin(6*pi*x)"}
            }
        }
    }
    
    # Time the solve
    start_time = time.time()
    result = solve(case_spec)
    end_time = time.time()
    
    print(f"\nSolve completed in {end_time - start_time:.3f} seconds")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Solver: {result['solver_info']['ksp_type']} with {result['solver_info']['pc_type']}")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")

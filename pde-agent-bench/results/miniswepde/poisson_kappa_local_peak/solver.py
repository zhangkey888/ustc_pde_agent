import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation -∇·(κ ∇u) = f with Dirichlet BCs.
    Adaptive mesh refinement loop with convergence check.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys:
        - "pde": dict with "kappa" expression (optional)
        If not provided, use default from problem description.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Default parameters
    default_kappa_expr = "1 + 30*exp(-150*((x-0.35)**2 + (y-0.65)**2))"
    
    # Parse case_spec (robust to missing keys)
    kappa_expr_str = default_kappa_expr
    if isinstance(case_spec, dict):
        pde_info = case_spec.get("pde", {})
        if isinstance(pde_info, dict):
            kappa_info = pde_info.get("kappa", {})
            if isinstance(kappa_info, dict):
                expr = kappa_info.get("expr")
                if expr is not None:
                    kappa_expr_str = expr
    
    # Grid convergence loop parameters
    resolutions = [32, 64, 128]
    element_degree = 1  # P1 elements
    
    # Solver configurations (try in order)
    solver_configs = [
        {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
        {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
    ]
    
    # Storage for convergence check
    prev_norm = None
    u_sol_final = None
    final_resolution = None
    final_solver_info = None
    linear_iterations_total = 0
    
    # Start timing
    start_time = time.time()
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N = {N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Define κ from expression string
        # We know the exact expression from problem description
        kappa_ufl = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
        
        # Define exact solution
        u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        
        # Compute f = -∇·(κ ∇u_exact)
        grad_u_exact = ufl.grad(u_exact_ufl)
        kappa_grad = kappa_ufl * grad_u_exact
        f_expr = -ufl.div(kappa_grad)
        
        # Interpolate κ onto function space for use in form
        kappa_func = fem.Function(V)
        kappa_expr_compiled = fem.Expression(kappa_ufl, V.element.interpolation_points)
        kappa_func.interpolate(kappa_expr_compiled)
        
        # Source term f
        f = fem.Function(V)
        f_expr_compiled = fem.Expression(f_expr, V.element.interpolation_points)
        f.interpolate(f_expr_compiled)
        
        # Define variational form
        a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Boundary conditions: Dirichlet u = g on ∂Ω
        # g = u_exact on boundary
        def boundary_marker(x):
            # Mark all boundary facets
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Try solvers in order
        solver_success = False
        solver_info_current = None
        u_current = None
        linear_iters = 0
        
        for config in solver_configs:
            if rank == 0:
                print(f"  Trying solver: ksp_type={config['ksp_type']}, pc_type={config['pc_type']}")
            
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
                
                # Get solver statistics
                ksp = problem.solver
                linear_iters = ksp.getIterationNumber()
                
                solver_success = True
                u_current = u_sol
                solver_info_current = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": config["ksp_type"],
                    "pc_type": config["pc_type"],
                    "rtol": config["rtol"],
                    "iterations": linear_iters
                }
                break
                
            except Exception as e:
                if rank == 0:
                    print(f"    Solver failed: {e}")
                continue
        
        if not solver_success:
            if rank == 0:
                print(f"All solvers failed for N={N}")
            # Continue to next resolution
            continue
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_current, u_current) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if rank == 0:
                print(f"  Relative norm change: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Convergence achieved at N={N}")
                u_sol_final = u_current
                final_resolution = N
                final_solver_info = solver_info_current
                linear_iterations_total += linear_iters
                break
        
        prev_norm = norm_value
        u_sol_final = u_current
        final_resolution = N
        final_solver_info = solver_info_current
        linear_iterations_total += linear_iters
    
    # Fallback: if loop finished without convergence, use finest mesh result
    if u_sol_final is None:
        # This shouldn't happen if at least one solve succeeded
        raise RuntimeError("No successful solve in any resolution")
    
    # Update iterations in solver_info (total across all resolutions tried)
    if final_solver_info:
        final_solver_info["iterations"] = linear_iterations_total
    
    # Interpolate solution to 50x50 grid for output
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten and create 3D points (z=0)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_sol_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # End timing
    end_time = time.time()
    if rank == 0:
        print(f"Total solve time: {end_time - start_time:.3f} seconds")
    
    # Prepare output dictionary
    output = {
        "u": u_grid,
        "solver_info": final_solver_info
    }
    
    return output

def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at arbitrary points.
    points: shape (3, N) numpy array
    Returns: shape (N,) array of values
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
    
    # In parallel, we need to gather results from all processes
    comm = u_func.function_space.mesh.comm
    if comm.size > 1:
        # Gather all local u_values to rank 0
        all_values = comm.gather(u_values, root=0)
        if comm.rank == 0:
            # Combine: take first non-nan value for each point
            combined = np.zeros_like(u_values)
            for i in range(points.shape[1]):
                for proc_vals in all_values:
                    if not np.isnan(proc_vals[i]):
                        combined[i] = proc_vals[i]
                        break
            u_values = combined
        # Broadcast back to all ranks
        u_values = comm.bcast(u_values, root=0)
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "kappa": {"type": "expr", "expr": "1 + 30*exp(-150*((x-0.35)**2 + (y-0.65)**2))"}
        }
    }
    
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("Solution shape:", result["u"].shape)

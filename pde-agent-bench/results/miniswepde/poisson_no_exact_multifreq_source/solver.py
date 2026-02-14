import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    convergence_tol = 0.01  # 1% relative error tolerance
    
    # Initialize variables for convergence check
    prev_norm = None
    u_sol_final = None
    domain_final = None
    mesh_resolution_used = None
    element_degree = 1  # Using linear elements
    solver_info = {}
    
    # Try iterative solver first, fallback to direct if fails
    solver_strategies = [
        {"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
        {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-12}
    ]
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet: u = 0 on entire boundary)
        # Since exact solution is not provided, we use homogeneous Dirichlet
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
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.sin(5 * np.pi * x[0]) * ufl.sin(3 * np.pi * x[1]) + \
                 0.5 * ufl.sin(9 * np.pi * x[0]) * ufl.sin(7 * np.pi * x[1])
        
        # Coefficient kappa = 1.0
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Try solver strategies
        problem = None
        u_sol = fem.Function(V)
        solver_success = False
        iterations = 0
        current_petsc_opts = {}
        
        for strategy_idx, petsc_opts in enumerate(solver_strategies):
            try:
                # Create linear problem with current solver strategy
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options=petsc_opts,
                    petsc_options_prefix=f"pdebench_{N}_"
                )
                u_sol = problem.solve()
                solver_success = True
                current_petsc_opts = petsc_opts
                
                # Get iteration count if available
                if hasattr(problem, '_solver'):
                    ksp = problem._solver
                    iterations = ksp.getIterationNumber()
                break
            except Exception as e:
                if strategy_idx == len(solver_strategies) - 1:
                    # Last strategy also failed, re-raise
                    raise
                # Try next strategy
                continue
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 0
            if relative_error < convergence_tol:
                # Converged
                u_sol_final = u_sol
                domain_final = domain
                mesh_resolution_used = N
                # Use the successful solver strategy info
                solver_info.update({
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": current_petsc_opts.get("ksp_type", "gmres"),
                    "pc_type": current_petsc_opts.get("pc_type", "hypre"),
                    "rtol": current_petsc_opts.get("ksp_rtol", 1e-8),
                    "iterations": iterations
                })
                break
        
        prev_norm = norm_value
        u_sol_final = u_sol
        domain_final = domain
        mesh_resolution_used = N
        # Store solver info for this resolution (will be overwritten if converges earlier)
        solver_info.update({
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": current_petsc_opts.get("ksp_type", "gmres"),
            "pc_type": current_petsc_opts.get("pc_type", "hypre"),
            "rtol": current_petsc_opts.get("ksp_rtol", 1e-8),
            "iterations": iterations
        })
    
    # If loop finished without convergence, use the finest mesh result
    if u_sol_final is None:
        # This shouldn't happen, but as safety
        u_sol_final = u_sol
        domain_final = domain
        mesh_resolution_used = resolutions[-1]
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    # z-coordinate is 0 for 2D
    
    # Evaluate function at points
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
    
    # All-gather results across MPI processes
    u_values_all = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
    # Replace NaN with 0 (points not found on any process)
    u_values_all = np.nan_to_num(u_values_all, nan=0.0)
    
    # Reshape to (nx, ny)
    u_grid = u_values_all.reshape((nx, ny))
    
    # Return result dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    result = solve(case_spec)
    print("Solver executed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")

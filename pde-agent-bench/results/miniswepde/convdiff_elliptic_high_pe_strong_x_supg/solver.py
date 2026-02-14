import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation: -ε ∇²u + β·∇u = f with SUPG stabilization.
    Adaptive mesh refinement with convergence check.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec
    epsilon = case_spec.get('epsilon', 0.01)
    beta_list = case_spec.get('beta', [15.0, 0.0])
    beta_array = np.array(beta_list, dtype=np.float64)
    
    # Domain: unit square
    domain_bounds = [[0.0, 0.0], [1.0, 1.0]]
    
    # Adaptive mesh refinement loop
    resolutions = [16, 32, 64]
    element_degree = 1  # Linear elements
    
    # For convergence tracking
    prev_error = None  # Track error instead of norm for accuracy
    u_sol_final = None
    final_resolution = None
    solver_info = {}
    domain_final = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_rectangle(comm, domain_bounds, [N, N], 
                                       cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define manufactured solution
        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        
        # Create constant vector for beta
        beta = fem.Constant(domain, beta_array)
        
        # Compute source term f from exact solution
        # -ε ∇²u_exact + β·∇u_exact = f
        laplacian_u = ufl.div(ufl.grad(u_exact))
        grad_u = ufl.grad(u_exact)
        f_expr = -epsilon * laplacian_u + ufl.dot(beta, grad_u)
        
        # Test and trial functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Standard Galerkin terms
        a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a += ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # SUPG stabilization parameter (tau)
        # Characteristic cell size
        h = ufl.CellVolume(domain)**(1.0/domain.topology.dim)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        # Standard SUPG parameter for linear elements
        tau = h / (2.0 * beta_norm)
        
        # Add SUPG stabilization terms
        a += tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
        L += tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx
        
        # Boundary conditions: Dirichlet using exact solution
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # All boundaries
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
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver with optimized settings
        try:
            # Use GMRES with ilu preconditioner
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options_prefix="cd_",
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "ilu",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                }
            )
            u_sol = problem.solve()
            ksp_type = "gmres"
            pc_type = "ilu"
            rtol = 1e-8
            # Get iteration count
            ksp = problem.solver
            iterations = ksp.getIterationNumber()
            
        except Exception as e:
            # Fallback to direct solver
            if rank == 0:
                print(f"Iterative solver failed: {e}. Switching to direct solver.")
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options_prefix="cd_direct_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "ksp_rtol": 1e-12
                }
            )
            u_sol = problem.solve()
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-12
            iterations = 1
        
        # Compute L2 error against exact solution
        error_expr = ufl.inner(u_sol - u_exact, u_sol - u_exact)
        error_form = fem.form(error_expr * ufl.dx)
        error_local = fem.assemble_scalar(error_form)
        error_global = comm.allreduce(error_local, op=MPI.SUM)
        current_error = np.sqrt(error_global)
        
        # Check convergence based on error change
        if prev_error is not None:
            error_change = abs(current_error - prev_error) / (prev_error + 1e-12)
            if error_change < 0.01:  # 1% change in error
                u_sol_final = u_sol
                final_resolution = N
                domain_final = domain
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": iterations,
                    "stabilization": "SUPG",
                    "converged_at_resolution": N
                }
                break
        
        prev_error = current_error
        u_sol_final = u_sol
        final_resolution = N
        domain_final = domain
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "stabilization": "SUPG",
            "converged_at_resolution": N if prev_error is None else "not_converged"
        }
    
    # If loop finished without break, use the finest mesh result
    if final_resolution is None:
        final_resolution = resolutions[-1]
        solver_info["converged_at_resolution"] = f"fallback_{final_resolution}"
    
    # Sample solution on 50x50 grid for output
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D coordinates)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
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
    
    # Gather all values to rank 0
    if comm.size > 1:
        all_values = comm.gather(u_values, root=0)
        if rank == 0:
            # Combine values from all processes
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        else:
            u_values = None
        u_values = comm.bcast(u_values, root=0)
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # End timing
    end_time = time.time()
    solver_info["wall_time_sec"] = end_time - start_time
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

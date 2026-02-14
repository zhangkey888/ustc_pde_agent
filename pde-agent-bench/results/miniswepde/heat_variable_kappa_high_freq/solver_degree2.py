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
    Returns solution on a 50x50 uniform grid and solver info.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters from case_spec
    # Handle different possible structures of case_spec
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', 0.1)
        dt_suggested = time_spec.get('dt', 0.005)
        scheme = time_spec.get('scheme', 'backward_euler')
    else:
        # Direct parameters (for backward compatibility)
        t_end = case_spec.get('t_end', 0.1)
        dt_suggested = case_spec.get('dt', 0.005)
        scheme = case_spec.get('scheme', 'backward_euler')
    
    # Force transient flag as per instructions
    is_transient = True
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]  # Progressive refinement
    element_degree = 2  # Quadratic elements (P2)
    
    # Solver parameters (adaptive)
    ksp_type = 'gmres'
    pc_type = 'hypre'
    rtol = 1e-8
    
    # Time-stepping parameters - start with suggested, may adapt
    dt = dt_suggested
    
    # Store convergence history
    norm_history = []
    solution_history = []
    mesh_history = []
    
    # Adaptive mesh refinement loop
    converged = False
    final_solution = None
    final_mesh_resolution = None
    total_linear_iterations = 0
    time_steps_computed = 0
    actual_dt_used = dt
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define manufactured solution and coefficients
        x = ufl.SpatialCoordinate(domain)
        
        # Exact solution: u_exact = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
        u_exact_expr = ufl.exp(-t_end) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        
        # Diffusion coefficient κ = 1 + 0.3*sin(6*pi*x)*sin(6*pi*y)
        kappa_expr = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
        
        # Source term f derived from exact solution: f = du/dt - ∇·(κ ∇u)
        u_exact = u_exact_expr
        du_dt = -u_exact  # derivative of exp(-t) is -exp(-t)
        grad_u = ufl.grad(u_exact)
        flux = kappa_expr * grad_u
        div_flux = ufl.div(flux)
        f_expr = du_dt - div_flux
        
        # Create boundary condition using the exact solution
        el = V.element
        interpolation_points = el.interpolation_points
        
        # Create expression for boundary condition
        u_bc_expr = fem.Expression(u_exact_expr, interpolation_points)
        u_bc_func = fem.Function(V)
        u_bc_func.interpolate(u_bc_expr)
        
        # Apply Dirichlet BC on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Find all boundary facets
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        # Locate DOFs on boundary
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create Dirichlet BC
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Solution at previous time step
        
        # Initial condition: u(x,0) = sin(2*pi*x)*sin(2*pi*y)
        u0_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        u0_interp = fem.Expression(u0_expr, interpolation_points)
        u_n.interpolate(u0_interp)
        
        # Define variational problem for backward Euler
        u = ufl.TrialFunction(V)  # Unknown at new time step
        v = ufl.TestFunction(V)
        dt_constant = fem.Constant(domain, ScalarType(dt))
        
        # Weak form for backward Euler:
        a = (1/dt_constant) * ufl.inner(u, v) * ufl.dx + ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1/dt_constant) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time)
        try:
            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            if rank == 0:
                print(f"  Matrix assembled, size: {A.getSize()}")
        except Exception as e:
            if rank == 0:
                print(f"  Matrix assembly failed: {e}")
            continue
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)  # Solution function
        
        # Setup linear solver with iterative first, direct fallback
        linear_iterations_this_mesh = 0
        solver_success = False
        
        # Time-stepping loop
        current_dt = dt
        t = 0.0
        steps = 0
        max_steps = int(t_end / dt) + 20  # Allow some extra
        
        # Track if we need to switch to direct solver
        use_direct_solver = False
        
        while t < t_end - 1e-10 and steps < max_steps:
            # Adjust dt for final step
            if t + current_dt > t_end:
                current_dt = t_end - t
                dt_constant.value = current_dt
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Create solver for this step (or reuse)
            if steps == 0 or use_direct_solver:
                ksp_solver = PETSc.KSP().create(domain.comm)
                ksp_solver.setOperators(A)
                
                if use_direct_solver:
                    ksp_solver.setType('preonly')
                    ksp_solver.getPC().setType('lu')
                else:
                    ksp_solver.setType('gmres')
                    ksp_solver.getPC().setType('hypre')
                    ksp_solver.setTolerances(rtol=rtol, max_it=1000)
            
            # Solve linear system
            try:
                ksp_solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                its = ksp_solver.getIterationNumber()
                linear_iterations_this_mesh += its
                
                # Check for NaN
                if np.any(np.isnan(u_sol.x.array)):
                    raise RuntimeError("Solution contains NaN")
                
                # Update for next step
                u_n.x.array[:] = u_sol.x.array
                t += current_dt
                steps += 1
                solver_success = True
                
            except Exception as e:
                if not use_direct_solver:
                    if rank == 0:
                        print(f"  Iterative solver failed at step {steps}, switching to direct solver")
                    use_direct_solver = True
                    continue  # Retry with direct solver
                else:
                    if rank == 0:
                        print(f"  Direct solver also failed at step {steps}: {e}")
                    break
        
        if not solver_success or steps == 0:
            if rank == 0:
                print(f"  Time-stepping failed for N={N}, skipping")
            continue
        
        total_linear_iterations += linear_iterations_this_mesh
        time_steps_computed = steps
        actual_dt_used = dt  # Could be adjusted if we implemented dt reduction
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        norm_history.append(norm_value)
        solution_history.append(u_sol.copy())
        mesh_history.append(N)
        
        if rank == 0:
            print(f"  Completed {steps} time steps, norm = {norm_value:.6e}, iterations = {linear_iterations_this_mesh}")
        
        # Check convergence (1% relative change in norm)
        if len(norm_history) >= 2:
            norm_old = norm_history[-2]
            norm_new = norm_history[-1]
            relative_error = abs(norm_new - norm_old) / norm_new if norm_new > 1e-15 else 0.0
            
            if rank == 0:
                print(f"  Relative error in norm: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                converged = True
                final_solution = u_sol
                final_mesh_resolution = N
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
    
    # Fallback: use finest mesh if not converged
    if not converged and solution_history:
        final_solution = solution_history[-1]
        final_mesh_resolution = mesh_history[-1]
        if rank == 0:
            print(f"  Using finest mesh N={final_mesh_resolution} (no convergence)")
    elif not solution_history:
        # Create dummy solution if everything failed
        domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        final_solution = fem.Function(V)
        final_mesh_resolution = 32
        if rank == 0:
            print("Warning: All resolutions failed, using dummy solution")
    
    # Interpolate solution to 50x50 uniform grid
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Create points array (3D even for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Get the domain from the final solution
    domain_final = final_solution.function_space.mesh
    
    # Build bounding box tree for point evaluation
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    
    # Find cells containing the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points.T)
    
    # Evaluate at points
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to rank 0
    u_all = comm.gather(u_values, root=0)
    
    if rank == 0:
        # Combine results from all processes
        u_combined = np.full(points.shape[1], np.nan, dtype=ScalarType)
        for proc_vals in u_all:
            mask = ~np.isnan(proc_vals)
            u_combined[mask] = proc_vals[mask]
        
        # Reshape to grid
        u_grid = u_combined.reshape(nx, ny)
    else:
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    # Broadcast grid to all ranks (for consistency)
    u_grid = comm.bcast(u_grid, root=0)
    
    # Also compute initial condition on grid
    V_final = final_solution.function_space
    u0_func = fem.Function(V_final)
    x = ufl.SpatialCoordinate(domain_final)
    u0_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Get interpolation points for this space
    el_final = V_final.element
    interpolation_points_final = el_final.interpolation_points
    u0_interp = fem.Expression(u0_expr, interpolation_points_final)
    u0_func.interpolate(u0_interp)
    
    # Evaluate initial condition on the same grid
    u0_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_all = comm.gather(u0_values, root=0)
    
    if rank == 0:
        u0_combined = np.full(points.shape[1], np.nan, dtype=ScalarType)
        for proc_vals in u0_all:
            mask = ~np.isnan(proc_vals)
            u0_combined[mask] = proc_vals[mask]
        u0_grid = u0_combined.reshape(nx, ny)
    else:
        u0_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    u0_grid = comm.bcast(u0_grid, root=0)
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": final_mesh_resolution if final_mesh_resolution else resolutions[-1],
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_linear_iterations,
        "dt": actual_dt_used,
        "n_steps": time_steps_computed,
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a simple case specification
    case_spec = {
        "t_end": 0.1,
        "dt": 0.005,
        "scheme": "backward_euler"
    }
    
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("Solution shape:", result["u"].shape)
    print("Initial condition shape:", result["u_initial"].shape)

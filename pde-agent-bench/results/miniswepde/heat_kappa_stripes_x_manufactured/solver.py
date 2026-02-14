import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    
    Problem: ∂u/∂t - ∇·(κ ∇u) = f in Ω × (0, T]
    with manufactured solution: u = exp(-t)*sin(2πx)*sin(πy)
    and κ = 1 + 0.5*sin(6πx)
    
    Returns:
        Dictionary with keys:
        - "u": solution array on 50x50 grid
        - "solver_info": dictionary with solver parameters and performance
        - "u_initial": (optional) initial condition on same grid
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters - use defaults from problem description
    t_end = 0.1
    dt = 0.01
    time_scheme = "backward_euler"
    
    # Check case_spec for time parameters (may be missing or incomplete)
    if case_spec is not None and 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # Adjust dt to exactly reach t_end
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    mesh_resolution_used = None
    element_degree = 1  # P1 elements
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Define κ as a UFL expression
        kappa_expr = 1.0 + 0.5 * ufl.sin(6 * ufl.pi * x[0])
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Solution at previous time step
        u_np1 = fem.Function(V)  # Solution at current time step
        
        # Initial condition (interpolate exact solution at t=0)
        u_n.interpolate(lambda x: np.exp(0) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
        
        # Boundary condition (Dirichlet, from exact solution on entire boundary)
        def boundary(x):
            return np.logical_or(
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0)
            ) | np.logical_or(
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            )
        
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        
        # Time-stepping loop
        total_linear_iterations = 0
        t = 0.0
        
        # Define forms for time-stepping
        # Backward Euler: (u - u_n)/dt - ∇·(κ ∇u) = f
        dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
        
        # κ as a function for the stiffness matrix
        kappa_func = fem.Function(V)
        kappa_func.interpolate(lambda x: 1.0 + 0.5*np.sin(6*np.pi*x[0]))
        
        # Mass matrix term
        m = u * v * ufl.dx
        # Stiffness matrix term
        a_kappa = kappa_func * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        
        # Left-hand side
        a = m + dt_constant * a_kappa
        a_form = fem.form(a)
        
        # Setup linear solver with iterative first, fallback to direct
        solver_success = False
        ksp_type_used = "gmres"
        pc_type_used = "hypre"
        rtol_used = 1e-8
        
        for solver_try in range(2):  # Try iterative, then direct
            try:
                # Time-stepping loop
                for step in range(n_steps):
                    t_prev = t
                    t += dt
                    
                    # Update boundary condition for current time
                    u_bc.interpolate(lambda x: np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
                    bc = fem.dirichletbc(u_bc, boundary_dofs)
                    
                    # Create source term f at current time t
                    # Analytical expression derived from manufactured solution:
                    # f = ∂u/∂t - ∇·(κ∇u)
                    # where u = exp(-t)*sin(2πx)*sin(πy), κ = 1 + 0.5*sin(6πx)
                    exp_term = ufl.exp(-t)
                    sin_pi_y = ufl.sin(ufl.pi * x[1])
                    sin_2pi_x = ufl.sin(2 * ufl.pi * x[0])
                    cos_2pi_x = ufl.cos(2 * ufl.pi * x[0])
                    cos_6pi_x = ufl.cos(6 * ufl.pi * x[0])
                    
                    f_expr = -exp_term * sin_pi_y * (
                        sin_2pi_x + 
                        6 * ufl.pi**2 * cos_6pi_x * cos_2pi_x - 
                        5 * ufl.pi**2 * kappa_expr * sin_2pi_x
                    )
                    
                    # Create RHS form
                    L = m * u_n + dt_constant * v * f_expr * ufl.dx
                    L_form = fem.form(L)
                    
                    # Assemble matrix with BCs
                    A = petsc.assemble_matrix(a_form, bcs=[bc])
                    A.assemble()
                    
                    # Create vectors
                    b = petsc.create_vector(L_form.function_spaces)
                    u_sol_vec = u_np1.x.petsc_vec
                    
                    # Create linear solver
                    ksp = PETSc.KSP().create(domain.comm)
                    ksp.setOperators(A)
                    
                    if solver_try == 0:
                        # Try iterative solver first
                        ksp.setType(PETSc.KSP.Type.GMRES)
                        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
                        ksp.setTolerances(rtol=rtol_used, atol=1e-12, max_it=1000)
                        ksp_type_used = "gmres"
                        pc_type_used = "hypre"
                    else:
                        # Fallback to direct solver
                        ksp.setType(PETSc.KSP.Type.PREONLY)
                        ksp.getPC().setType(PETSc.PC.Type.LU)
                        ksp_type_used = "preonly"
                        pc_type_used = "lu"
                    
                    ksp.setFromOptions()
                    
                    # Assemble RHS
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting for Dirichlet BC (modifies RHS for non-homogeneous BCs)
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    
                    # Apply BCs to RHS (sets Dirichlet DOFs to BC values)
                    petsc.set_bc(b, [bc])
                    
                    # Solve linear system
                    ksp.solve(b, u_sol_vec)
                    total_linear_iterations += ksp.getIterationNumber()
                    
                    # Update solution
                    u_np1.x.scatter_forward()
                    
                    # Prepare for next step
                    u_n.x.array[:] = u_np1.x.array
                    
                solver_success = True
                break
                
            except Exception as e:
                if rank == 0:
                    print(f"Solver try {solver_try} failed: {e}")
                if solver_try == 1:  # Direct solver also failed
                    raise
        
        if not solver_success:
            raise RuntimeError("All solver attempts failed")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_np1, u_np1) * ufl.dx)
        norm_value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        norms.append(norm_value)
        solutions.append((domain, V, u_np1))
        
        # Check convergence (1% relative change in norm)
        if len(norms) >= 2:
            relative_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 1e-12 else 0.0
            if rank == 0:
                print(f"  Relative error in norm: {relative_error:.6f}")
            if relative_error < 0.01:  # 1% convergence criterion
                mesh_resolution_used = N
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
    
    # Use the last solution if convergence not reached
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
        if rank == 0:
            print(f"  Using fallback resolution N={mesh_resolution_used}")
    
    # Get the final solution
    final_idx = resolutions.index(mesh_resolution_used) if mesh_resolution_used in resolutions else -1
    final_domain, final_V, final_u = solutions[final_idx]
    
    # Sample solution on 50x50 grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points for evaluation (3D format required)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points.T)
    
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
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to rank 0 in parallel runs
    if comm.size > 1:
        all_u_values = comm.gather(u_values, root=0)
        if rank == 0:
            # Combine values from all processes
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_u_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        else:
            u_values = None
        u_values = comm.bcast(u_values, root=0)
    
    u_grid = u_values.reshape((nx, ny))
    
    # Also get initial condition on the same grid for optional output
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: np.exp(0) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    # Evaluate initial condition
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    if comm.size > 1:
        all_u0_values = comm.gather(u0_values, root=0)
        if rank == 0:
            combined0 = np.full_like(u0_values, np.nan)
            for proc_vals in all_u0_values:
                mask = ~np.isnan(proc_vals)
                combined0[mask] = proc_vals[mask]
            u0_values = combined0
        else:
            u0_values = None
        u0_values = comm.bcast(u0_values, root=0)
    
    u_initial = u0_values.reshape((nx, ny)) if u0_values is not None else None
    
    # Prepare solver_info with all required fields
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    if u_initial is not None:
        result["u_initial"] = u_initial
    
    return result

if __name__ == "__main__":
    # Simple test to verify the solver runs
    import time
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    
    start_time = time.time()
    result = solve(case_spec)
    end_time = time.time()
    
    print(f"\nSolver completed in {end_time - start_time:.3f} seconds")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Time steps: {result['solver_info']['n_steps']}")
    print(f"Linear iterations: {result['solver_info']['iterations']}")

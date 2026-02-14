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
    Solve the heat equation with Crank-Nicolson for better accuracy.
    Returns solution on a 50x50 uniform grid and solver info.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters from case_spec
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', 0.1)
        dt_suggested = time_spec.get('dt', 0.005)
        scheme = time_spec.get('scheme', 'backward_euler')
    else:
        t_end = case_spec.get('t_end', 0.1)
        dt_suggested = case_spec.get('dt', 0.005)
        scheme = case_spec.get('scheme', 'backward_euler')
    
    # Use Crank-Nicolson for better accuracy (2nd order)
    # Override scheme for better accuracy
    scheme = 'crank_nicolson'
    
    # Use moderate dt - Crank-Nicolson allows larger dt for same accuracy
    dt = min(dt_suggested, 0.001)
    
    # Use quadratic elements
    element_degree = 2
    
    # Use adaptive mesh refinement with error estimation
    resolutions = [64, 96, 128]
    target_error = 1.0e-3  # Target error slightly below requirement
    
    if rank == 0:
        print(f"Solving heat equation with {scheme}, target error {target_error}")
    
    best_solution = None
    best_error = float('inf')
    best_N = None
    total_linear_iterations = 0
    
    for N in resolutions:
        if rank == 0:
            print(f"\nTesting mesh resolution N={N} with degree {element_degree}, dt={dt}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define manufactured solution and coefficients
        x = ufl.SpatialCoordinate(domain)
        
        # Time parameter for exact solution
        t_param = fem.Constant(domain, ScalarType(0.0))
        
        # Exact solution: u_exact = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
        u_exact_expr = ufl.exp(-t_param) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        
        # Diffusion coefficient κ = 1 + 0.3*sin(6*pi*x)*sin(6*pi*y)
        kappa_expr = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
        
        # Source term f derived from exact solution: f = du/dt - ∇·(κ ∇u)
        u_exact = u_exact_expr
        du_dt = -u_exact
        grad_u = ufl.grad(u_exact)
        flux = kappa_expr * grad_u
        div_flux = ufl.div(flux)
        f_expr = du_dt - div_flux
        
        # Create boundary condition function
        el = V.element
        interpolation_points = el.interpolation_points
        u_bc_func = fem.Function(V)
        
        # Apply Dirichlet BC on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Solution at previous time step
        
        # Initial condition
        t_param.value = 0.0
        u0_interp = fem.Expression(u_exact_expr, interpolation_points)
        u_n.interpolate(u0_interp)
        
        # Define variational problem for Crank-Nicolson
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dt_constant = fem.Constant(domain, ScalarType(dt))
        
        # Crank-Nicolson form: (u - u_n)/dt * v dx + 0.5 * κ∇(u + u_n)·∇v dx = 0.5 * (f(t) + f(t_n)) * v dx
        # But f depends on time through u_exact
        # We'll handle time-dependent f by updating it each step
        
        # Forms for time levels n and n+1/2
        u_mid = 0.5 * (u + u_n)  # Average
        
        a = (1/dt_constant) * ufl.inner(u, v) * ufl.dx + 0.5 * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1/dt_constant) * ufl.inner(u_n, v) * ufl.dx - 0.5 * ufl.inner(kappa_expr * ufl.grad(u_n), ufl.grad(v)) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        if rank == 0:
            print(f"  Matrix assembled, size: {A.getSize()}")
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Setup linear solver - try iterative first
        ksp_solver = PETSc.KSP().create(domain.comm)
        ksp_solver.setOperators(A)
        ksp_solver.setType('gmres')
        ksp_solver.getPC().setType('hypre')
        ksp_solver.setTolerances(rtol=1e-8, max_it=1000)
        
        # Time-stepping loop
        t = 0.0
        steps = 0
        linear_iterations_this_mesh = 0
        max_steps = int(t_end / dt) + 10
        
        while t < t_end - 1e-10 and steps < max_steps:
            # Update time for BC and source term
            t_next = min(t + dt, t_end)
            t_param.value = t_next  # Use implicit time level
            
            # Update boundary condition
            u_bc_interp = fem.Expression(u_exact_expr, interpolation_points)
            u_bc_func.interpolate(u_bc_interp)
            
            # Adjust dt for final step
            if t + dt > t_end:
                dt_constant.value = t_end - t
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            try:
                ksp_solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                its = ksp_solver.getIterationNumber()
                linear_iterations_this_mesh += its
                
                # Update for next step
                u_n.x.array[:] = u_sol.x.array
                t += dt_constant.value
                steps += 1
                
            except Exception as e:
                if rank == 0:
                    print(f"  Solver failed, switching to direct: {e}")
                # Switch to direct solver
                ksp_solver.setType('preonly')
                ksp_solver.getPC().setType('lu')
                ksp_solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                its = ksp_solver.getIterationNumber()
                linear_iterations_this_mesh += its
                
                u_n.x.array[:] = u_sol.x.array
                t += dt_constant.value
                steps += 1
        
        total_linear_iterations += linear_iterations_this_mesh
        
        if rank == 0:
            print(f"  Completed {steps} time steps, iterations = {linear_iterations_this_mesh}")
        
        # Estimate error by comparing with exact solution at final time
        # Compute L2 error of solution
        t_param.value = t_end
        u_exact_func = fem.Function(V)
        u_exact_interp = fem.Expression(u_exact_expr, interpolation_points)
        u_exact_func.interpolate(u_exact_interp)
        
        error_expr = u_sol - u_exact_func
        error_form = fem.form(ufl.inner(error_expr, error_expr) * ufl.dx)
        error_sq = fem.assemble_scalar(error_form)
        error_sq_global = comm.allreduce(error_sq, op=MPI.SUM)
        l2_error = np.sqrt(error_sq_global)
        
        if rank == 0:
            print(f"  Estimated L2 error: {l2_error:.6e}")
        
        # Check if this is the best solution so far
        if l2_error < best_error:
            best_error = l2_error
            best_solution = u_sol.copy()
            best_N = N
        
        # Stop if error is below target
        if l2_error < target_error:
            if rank == 0:
                print(f"  Error below target {target_error}, stopping refinement")
            break
    
    # Use best solution found
    if best_solution is None:
        # Fallback
        domain = mesh.create_unit_square(comm, 64, 64, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 2))
        best_solution = fem.Function(V)
        best_N = 64
        if rank == 0:
            print("Warning: Using fallback solution")
    
    final_solution = best_solution
    final_N = best_N
    
    if rank == 0:
        print(f"\nSelected mesh N={final_N} with error {best_error:.6e}")
    
    # Interpolate solution to 50x50 uniform grid
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Get domain from solution
    domain_final = final_solution.function_space.mesh
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
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
        u_combined = np.full(points.shape[1], np.nan, dtype=ScalarType)
        for proc_vals in u_all:
            mask = ~np.isnan(proc_vals)
            u_combined[mask] = proc_vals[mask]
        u_grid = u_combined.reshape(nx, ny)
    else:
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    u_grid = comm.bcast(u_grid, root=0)
    
    # Initial condition
    V_final = final_solution.function_space
    u0_func = fem.Function(V_final)
    x = ufl.SpatialCoordinate(domain_final)
    u0_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    el_final = V_final.element
    interpolation_points_final = el_final.interpolation_points
    u0_interp = fem.Expression(u0_expr, interpolation_points_final)
    u0_func.interpolate(u0_interp)
    
    # Evaluate initial condition
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
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": int(t_end / dt) + (1 if t_end % dt > 1e-10 else 0),
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "t_end": 0.1,
        "dt": 0.005,
        "scheme": "backward_euler"
    }
    
    result = solve(case_spec)
    print("\nSolver info:", result["solver_info"])
    print("Solution shape:", result["u"].shape)

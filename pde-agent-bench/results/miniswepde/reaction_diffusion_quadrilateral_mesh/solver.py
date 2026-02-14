import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve reaction-diffusion equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract parameters
    pde_info = case_spec.get('pde', {})
    time_info = pde_info.get('time', {})
    
    # Force transient if t_end or dt is mentioned in problem description
    # According to problem description: t_end = 0.4, dt = 0.01
    is_transient = True  # This problem is transient
    t_end = 0.4
    dt_suggested = 0.01
    scheme = 'backward_euler'
    
    # Override with case_spec if provided
    if time_info:
        t_end = time_info.get('t_end', t_end)
        dt_suggested = time_info.get('dt', dt_suggested)
        scheme = time_info.get('scheme', scheme)
    
    # Reaction term parameters
    reaction_info = pde_info.get('reaction', {})
    reaction_type = reaction_info.get('type', 'linear')
    alpha = reaction_info.get('alpha', 1.0)
    
    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    element_degree = 2  # Start with degree 2 for accuracy
    
    u_solutions = []
    norms = []
    solver_info_list = []
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_rectangle(
            comm, 
            [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
            [N, N], 
            cell_type=mesh.CellType.quadrilateral
        )
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Exact solution for manufactured problem
        x = ufl.SpatialCoordinate(domain)
        t_var = fem.Constant(domain, ScalarType(0.0))
        u_exact = ufl.exp(-t_var) * (ufl.exp(x[0]) * ufl.sin(np.pi * x[1]))
        
        # Parameters
        epsilon = 1.0  # diffusion coefficient
        
        # Compute source term f from exact solution
        du_dt = -u_exact  # derivative of exp(-t) is -exp(-t)
        laplacian_u = ufl.div(ufl.grad(u_exact))
        
        # Reaction term
        if reaction_type == 'cubic':
            R_u = alpha * u_exact**3
        else:  # linear or none
            R_u = ScalarType(0.0) * u_exact
        
        # Source term: ∂u/∂t - ε∇²u + R(u) = f
        f_expr = du_dt - epsilon * laplacian_u + R_u
        
        # Boundary condition (Dirichlet)
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)  # All boundaries
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Initialize variables for solver info
        total_linear_iters = 0
        nonlinear_iters = []
        n_steps = 0
        dt_used = dt_suggested
        
        if is_transient:
            # Time-stepping
            dt = fem.Constant(domain, ScalarType(dt_suggested))
            u_n = fem.Function(V)  # Solution at time n
            u_prev = fem.Function(V)  # Solution at time n-1
            
            # Initial condition
            t_var.value = 0.0
            u_prev.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
            u_n.x.array[:] = u_prev.x.array[:]
            
            # Time-stepping loop
            t = 0.0
            n_steps = int(np.ceil(t_end / dt_suggested))
            dt_used = dt_suggested
            
            # Function for source term
            f_func = fem.Function(V)
            
            if reaction_type == 'cubic':
                # Nonlinear problem
                u_nl = fem.Function(V)
                u_nl.x.array[:] = u_prev.x.array[:]
                
                # Define nonlinear form
                F = (1/dt) * ufl.inner(u_nl - u_prev, v) * ufl.dx + \
                    epsilon * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * ufl.dx + \
                    alpha * ufl.inner(u_nl**3, v) * ufl.dx - \
                    ufl.inner(f_func, v) * ufl.dx
                
                # Create nonlinear problem
                problem = petsc.NewtonSolverNonlinearProblem(F, u_nl, bcs=[bc])
                newton_solver = NewtonSolver(domain.comm, problem)
                newton_solver.convergence_criterion = "incremental"
                newton_solver.rtol = 1e-8
                newton_solver.max_it = 20
                
                # Configure linear solver
                ksp = newton_solver.krylov_solver
                ksp.setType(PETSc.KSP.Type.GMRES)
                pc = ksp.getPC()
                pc.setType(PETSc.PC.Type.HYPRE)
                ksp.setTolerances(rtol=1e-8)
                
                # Time stepping
                for step in range(n_steps):
                    t += dt_used
                    t_var.value = t
                    
                    # Update boundary condition and source term
                    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
                    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
                    
                    # Solve nonlinear problem
                    n_iter, converged = newton_solver.solve(u_nl)
                    nonlinear_iters.append(n_iter)
                    total_linear_iters += newton_solver.krylov_solver.getIterationNumber()
                    
                    if not converged:
                        if rank == 0:
                            print(f"Warning: Newton solver did not converge at step {step}")
                    
                    # Update for next step
                    u_prev.x.array[:] = u_nl.x.array[:]
                
                final_u = u_nl
                
            else:
                # Linear problem
                # Define linear form
                F = (1/dt) * ufl.inner(u - u_prev, v) * ufl.dx + \
                    epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - \
                    ufl.inner(f_func, v) * ufl.dx
                
                a = ufl.lhs(F)
                L = ufl.rhs(F)
                
                # Create linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                    petsc_options_prefix="rd_"
                )
                
                # Time stepping
                for step in range(n_steps):
                    t += dt_used
                    t_var.value = t
                    
                    # Update boundary condition and source term
                    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
                    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
                    
                    # Solve linear problem
                    u_sol = problem.solve()
                    total_linear_iters += problem.solver.getIterationNumber()
                    
                    # Update for next step
                    u_prev.x.array[:] = u_sol.x.array[:]
                
                final_u = u_sol
                
        else:
            # Steady-state problem
            f_func = fem.Function(V)
            t_var.value = 0.0  # For steady, use t=0
            f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
            u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
            
            if reaction_type == 'cubic':
                # Nonlinear steady problem
                u_nl = fem.Function(V)
                u_nl.x.array[:] = 0.0  # Initial guess
                
                F = epsilon * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * ufl.dx + \
                    alpha * ufl.inner(u_nl**3, v) * ufl.dx - \
                    ufl.inner(f_func, v) * ufl.dx
                
                problem = petsc.NewtonSolverNonlinearProblem(F, u_nl, bcs=[bc])
                newton_solver = NewtonSolver(domain.comm, problem)
                newton_solver.convergence_criterion = "incremental"
                newton_solver.rtol = 1e-8
                newton_solver.max_it = 20
                
                ksp = newton_solver.krylov_solver
                ksp.setType(PETSc.KSP.Type.GMRES)
                pc = ksp.getPC()
                pc.setType(PETSc.PC.Type.HYPRE)
                ksp.setTolerances(rtol=1e-8)
                
                n_iter, converged = newton_solver.solve(u_nl)
                nonlinear_iters.append(n_iter)
                total_linear_iters = newton_solver.krylov_solver.getIterationNumber()
                final_u = u_nl
                
            else:
                # Linear steady problem
                F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - \
                    ufl.inner(f_func, v) * ufl.dx
                
                a = ufl.lhs(F)
                L = ufl.rhs(F)
                
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                    petsc_options_prefix="rd_"
                )
                
                final_u = problem.solve()
                total_linear_iters = problem.solver.getIterationNumber()
        
        # Compute norm for convergence check
        norm_form = fem.form(ufl.inner(final_u, final_u) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        norms.append(norm_value)
        u_solutions.append((domain, final_u, V))
        
        # Store solver info for this resolution
        solver_info_list.append({
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_linear_iters,
            "nonlinear_iterations": nonlinear_iters if nonlinear_iters else None,
            "dt": dt_used if is_transient else None,
            "n_steps": n_steps if is_transient else None,
            "time_scheme": scheme if is_transient else None
        })
        
        # Check convergence
        if len(norms) >= 2:
            rel_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 1e-12 else 1.0
            if rank == 0:
                print(f"Relative error at N={N}: {rel_error:.6f}")
            if rel_error < 0.01:
                if rank == 0:
                    print(f"Converged at N={N}, relative error={rel_error:.6f}")
                break
    
    # Use the last solution (either converged or finest mesh)
    domain, final_u, V = u_solutions[-1]
    solver_info = solver_info_list[-1]
    
    # Clean up solver_info
    if solver_info["nonlinear_iterations"] is None:
        del solver_info["nonlinear_iterations"]
    if not is_transient:
        del solver_info["dt"]
        del solver_info["n_steps"]
        del solver_info["time_scheme"]
    
    # Sample solution on 60x60 grid
    nx, ny = 60, 60
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather results from all processes
    u_all = comm.gather(u_values, root=0)
    if rank == 0:
        u_combined = np.concatenate([arr for arr in u_all if arr is not None])
        u_grid = u_combined.reshape(nx, ny)
    else:
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    u_grid = comm.bcast(u_grid, root=0)
    
    # Prepare result dictionary
    result = {"u": u_grid, "solver_info": solver_info}
    
    # Add initial condition for transient problems
    if is_transient:
        t_var.value = 0.0
        u_initial_func = fem.Function(V)
        u_initial_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        u_initial_vals = np.full((points.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_initial_vals[eval_map] = vals_init.flatten()
        
        u_init_all = comm.gather(u_initial_vals, root=0)
        if rank == 0:
            u_init_combined = np.concatenate([arr for arr in u_init_all if arr is not None])
            u_initial_grid = u_init_combined.reshape(nx, ny)
        else:
            u_initial_grid = np.zeros((nx, ny), dtype=ScalarType)
        
        u_initial_grid = comm.bcast(u_initial_grid, root=0)
        result["u_initial"] = u_initial_grid
    
    return result

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "time": {
                "t_end": 0.4,
                "dt": 0.01,
                "scheme": "backward_euler"
            },
            "reaction": {
                "type": "linear"
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Test completed successfully")
        print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
        print(f"Solution shape: {result['u'].shape}")

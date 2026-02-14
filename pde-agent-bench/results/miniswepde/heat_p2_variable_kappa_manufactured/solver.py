import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx import nls
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Start timing
    start_time = time.time()
    
    # Extract problem parameters
    # Force is_transient = True if t_end or dt mentioned in problem description
    # Problem description mentions t_end=0.06, dt=0.01
    t_end = 0.06
    dt = 0.01
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Manufactured solution for error checking (not used in solve, just for convergence test)
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    # Variable kappa expression
    def kappa_expr(x):
        return 1.0 + 0.4 * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with P1 elements
    
    # Storage for previous solution norm
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_u = None
    final_domain = None
    final_V = None
    
    # Solver statistics
    total_linear_iterations = 0
    nonlinear_iterations = []  # Not used for linear problem but keep for API
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define functions for current and previous time steps
        u_n = fem.Function(V)  # Previous time step
        u_ = fem.Function(V)   # Current time step (to be solved)
        
        # Initial condition
        def u0_expr(x):
            return exact_solution(x, 0.0)
        u_n.interpolate(u0_expr)
        
        # Variable kappa as a function
        kappa_func = fem.Function(V)
        kappa_func.interpolate(kappa_expr)
        
        # Time-stepping parameters
        n_steps = int(np.round(t_end / dt))
        if n_steps == 0:
            n_steps = 1
            dt = t_end
        
        # Define variational problem for backward Euler
        x = ufl.SpatialCoordinate(domain)
        t = fem.Constant(domain, ScalarType(0.0))
        
        # Manufactured source term f = du/dt - div(kappa * grad(u))
        u_exact = ufl.exp(-t) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        grad_u = ufl.grad(u_exact)
        div_kappa_grad_u = ufl.div(kappa_func * grad_u)
        f_expr = -u_exact - div_kappa_grad_u
        
        # Variational form
        a = (u * v / dt + kappa_func * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L = (u_n * v / dt + f_expr * v) * ufl.dx
        
        # Boundary conditions: Dirichlet using exact solution
        def boundary_marker(x):
            # Mark all boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function
        u_bc = fem.Function(V)
        def bc_expr(x):
            return exact_solution(x, t.value)
        u_bc.interpolate(bc_expr)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for this linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        for solver_try in range(2):  # Try iterative, then direct
            if solver_try == 0:
                # Iterative solver
                ksp = PETSc.KSP().create(domain.comm)
                ksp.setOperators(A)
                ksp.setType(PETSc.KSP.Type.GMRES)
                ksp.getPC().setType(PETSc.PC.Type.HYPRE)
                ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                ksp.setFromOptions()
            else:
                # Direct solver fallback
                ksp = PETSc.KSP().create(domain.comm)
                ksp.setOperators(A)
                ksp.setType(PETSc.KSP.Type.PREONLY)
                ksp.getPC().setType(PETSc.PC.Type.LU)
                ksp.setFromOptions()
                ksp_type = 'preonly'
                pc_type = 'lu'
            
            try:
                # Time-stepping loop
                step_iterations = 0
                for step in range(n_steps):
                    # Update time
                    t.value = (step + 1) * dt
                    
                    # Update boundary condition with current time
                    u_bc.interpolate(bc_expr)
                    
                    # Assemble RHS
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting and BCs
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Solve linear system
                    ksp.solve(b, u_sol.x.petsc_vec)
                    step_its = ksp.getIterationNumber()
                    step_iterations += step_its
                    
                    # Check for convergence failure
                    if ksp.getConvergedReason() <= 0:
                        if solver_try == 0:
                            if rank == 0:
                                print(f"  Iterative solver failed at step {step}, trying direct solver")
                            break  # Break to try direct solver
                        else:
                            raise RuntimeError("Direct solver also failed")
                    
                    # Update solution for next step
                    u_sol.x.scatter_forward()
                    u_n.x.array[:] = u_sol.x.array
                
                else:  # No break occurred, solver succeeded
                    solver_success = True
                    total_linear_iterations += step_iterations
                    break
                    
            except Exception as e:
                if solver_try == 0:
                    if rank == 0:
                        print(f"  Iterative solver failed: {e}, trying direct solver")
                    continue
                else:
                    raise
        
        if not solver_success:
            raise RuntimeError("Both iterative and direct solvers failed")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value
            if rank == 0:
                print(f"  Relative norm change: {relative_error:.6f}")
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Converged at N={N}")
                final_solution = u_sol
                final_mesh_resolution = N
                final_u = u_sol
                final_domain = domain
                final_V = V
                break
        
        prev_norm = norm_value
        final_solution = u_sol
        final_mesh_resolution = N
        final_u = u_sol
        final_domain = domain
        final_V = V
    
    # If loop finished without convergence, use finest mesh result
    if final_solution is None:
        final_solution = u_sol
        final_mesh_resolution = resolutions[-1]
        final_u = u_sol
        final_domain = domain
        final_V = V
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D format)
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
    
    # Gather all values to rank 0 (or use comm.gather for parallel)
    if comm.size > 1:
        all_values = comm.gather(u_values, root=0)
        if rank == 0:
            # Combine results from all processes
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        else:
            u_values = None
        u_values = comm.bcast(u_values, root=0)
    
    u_grid = u_values.reshape((nx, ny))
    
    # Also sample initial condition for optional output
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: exact_solution(x, 0.0))
    
    # Re-evaluate initial condition at same points
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        all_values = comm.gather(u0_values, root=0)
        if rank == 0:
            combined = np.full_like(u0_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u0_values = combined
        else:
            u0_values = None
        u0_values = comm.bcast(u0_values, root=0)
    
    u_initial = u0_values.reshape((nx, ny)) if u0_values is not None else None
    
    # Check if we're within time limit
    elapsed = time.time() - start_time
    if rank == 0:
        print(f"Total solver time: {elapsed:.3f}s")
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    # Add optional initial condition field
    if u_initial is not None:
        result["u_initial"] = u_initial
    
    return result

if __name__ == "__main__":
    # Test the solver with a minimal case specification
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.06,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Linear iterations: {result['solver_info']['iterations']}")

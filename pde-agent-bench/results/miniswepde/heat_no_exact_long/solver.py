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
    Solve transient heat equation with adaptive mesh refinement and runtime auto-tuning.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters with defaults
    # Problem description says t_end=0.2, dt=0.02, scheme=backward_euler
    # Force is_transient = True as per instructions
    t_end = 0.2
    dt = 0.02
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Domain is unit square [0,1]x[0,1]
    # Coefficients
    kappa = 0.8  # thermal diffusivity
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with P1 elements
    
    # Storage for convergence check
    prev_norm = None
    u_final = None
    mesh_resolution_used = None
    solver_info_final = None
    
    # Time tracking for overall solve
    total_start_time = time.time()
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        # Problem says u = g on ∂Ω, but g not specified. Assuming homogeneous Dirichlet.
        # Use topological approach
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
        
        # Homogeneous Dirichlet BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define functions for time-stepping
        u_n = fem.Function(V)  # Previous time step
        u_ = fem.Function(V)   # Current solution (to be computed)
        
        # Initial condition u0 = sin(2*pi*x)*sin(pi*y)
        def u0_expr(x):
            return np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
        
        u_n.interpolate(u0_expr)
        u_.interpolate(u0_expr)  # Initial guess
        
        # Source term f = cos(2*pi*x)*sin(pi*y)
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.cos(2*np.pi*x[0]) * ufl.sin(np.pi*x[1])
        f = fem.Constant(domain, ScalarType(1.0))  # Will be updated in form
        
        # Time-stepping scheme: backward Euler
        # (u - u_n)/dt * v * dx + kappa * dot(grad(u), grad(v)) * dx = f * v * dx
        a = (1/dt) * ufl.inner(u, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1/dt) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol_vec = u_.x.petsc_vec
        
        # Solver setup with robustness
        # Try iterative solver first, fallback to direct if fails
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Set iterative solver options
        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
        
        # Time-stepping loop
        n_steps = int(np.ceil(t_end / dt))
        actual_dt = t_end / n_steps  # Adjust dt to exactly reach t_end
        
        # Track solver iterations
        total_linear_iterations = 0
        
        try:
            for step in range(n_steps):
                # Update time level
                t = (step + 1) * actual_dt
                
                # Assemble RHS
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                
                # Apply lifting and BCs
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Solve linear system
                solver.solve(b, u_sol_vec)
                u_.x.scatter_forward()
                
                # Get iteration count
                its = solver.getIterationNumber()
                total_linear_iterations += its
                
                # Update previous solution
                u_n.x.array[:] = u_.x.array[:]
            
            # Success with iterative solver
            ksp_type = "gmres"
            pc_type = "hypre"
            
        except Exception as e:
            if rank == 0:
                print(f"Iterative solver failed: {e}. Switching to direct solver.")
            
            # Fallback to direct solver
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            
            # Reset and re-run time steps
            u_n.interpolate(u0_expr)
            u_.interpolate(u0_expr)
            total_linear_iterations = 0
            
            for step in range(n_steps):
                t = (step + 1) * actual_dt
                
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                solver.solve(b, u_sol_vec)
                u_.x.scatter_forward()
                
                its = solver.getIterationNumber()
                total_linear_iterations += its
                
                u_n.x.array[:] = u_.x.array[:]
            
            ksp_type = "preonly"
            pc_type = "lu"
        
        # Compute norm for convergence check
        norm = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_, u_) * ufl.dx)), op=MPI.SUM))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 1e-12 else abs(norm - prev_norm)
            if rank == 0:
                print(f"  Relative error: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Converged at N={N}")
                u_final = u_
                mesh_resolution_used = N
                # Store solver info
                solver_info_final = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": 1e-8,
                    "iterations": total_linear_iterations,
                    "dt": actual_dt,
                    "n_steps": n_steps,
                    "time_scheme": time_scheme
                }
                break
        
        prev_norm = norm
        
        # Store last result in case we finish loop without convergence
        u_final = u_
        mesh_resolution_used = N
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme
        }
    
    # If loop completed without break (no convergence), use finest mesh result
    if u_final is None:
        # This shouldn't happen but as safety
        u_final = u_
        mesh_resolution_used = 128
    
    # Prepare output on a 50x50 uniform grid
    # Create evaluation points
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten and create 3D points (z=0 for 2D)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_values = evaluate_function_at_points(u_final, points)
    u_grid = u_values.reshape(nx, ny)
    
    # Evaluate initial condition at same points for u_initial
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    u0_values = evaluate_function_at_points(u0_func, points)
    u_initial_grid = u0_values.reshape(nx, ny)
    
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Total solve time: {total_time:.2f}s")
        print(f"Mesh resolution used: {mesh_resolution_used}")
        print(f"Solver iterations: {solver_info_final['iterations']}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info_final
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at arbitrary points.
    points: shape (3, N) numpy array
    Returns: shape (N,) numpy array of values
    """
    comm = u_func.function_space.mesh.comm
    domain = u_func.function_space.mesh
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells containing points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build lists of points that are found on this processor
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Initialize result array with NaN
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    
    if len(points_on_proc) > 0:
        # Evaluate function at found points
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather results from all processors (in parallel)
    # Simple approach: just use values from root for output
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Gather all values to root
    if comm.size > 1:
        # Collect all indices and values
        all_indices = comm.gather(eval_map, root=0)
        all_values = comm.gather(vals.flatten() if len(points_on_proc) > 0 else np.array([], dtype=ScalarType), root=0)
        
        if rank == 0:
            # Combine results from all processors
            for proc_idx, (indices, values) in enumerate(zip(all_indices, all_values)):
                if len(indices) > 0:
                    u_values[indices] = values
        # Broadcast final array from root
        comm.Bcast(u_values, root=0)
    
    return u_values


# For testing when run directly
if __name__ == "__main__":
    # Test with minimal case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.2,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Test completed successfully")
    print(f"u shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

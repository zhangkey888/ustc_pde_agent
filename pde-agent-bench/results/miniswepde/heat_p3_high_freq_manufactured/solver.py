import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry, nls, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

# Define the scalar type
ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    # For transient problems, ensure time parameters exist
    t_end = 0.08
    dt_suggested = 0.008
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Force is_transient = True since this is a heat equation
    is_transient = True
    
    # Manufactured solution
    def u_exact(x, t):
        """u = exp(-t)*sin(3*pi*x)*sin(3*pi*y)"""
        return np.exp(-t) * np.sin(3*np.pi*x[0]) * np.sin(3*np.pi*x[1])
    
    def f_source(x, t):
        """Source term f derived from manufactured solution"""
        # ∂u/∂t - ∇·(κ ∇u) = f
        # u = exp(-t)*sin(3*pi*x)*sin(3*pi*y)
        # ∂u/∂t = -exp(-t)*sin(3*pi*x)*sin(3*pi*y)
        # ∇u = [3*pi*exp(-t)*cos(3*pi*x)*sin(3*pi*y), 3*pi*exp(-t)*sin(3*pi*x)*cos(3*pi*y)]
        # ∇·(κ ∇u) = κ * (∂²u/∂x² + ∂²u/∂y²)
        # ∂²u/∂x² = -9*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y)
        # ∂²u/∂y² = -9*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y)
        # ∇·(κ ∇u) = κ * (-18*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y))
        κ = 1.0
        u_val = np.exp(-t) * np.sin(3*np.pi*x[0]) * np.sin(3*np.pi*x[1])
        laplacian = -18 * np.pi**2 * u_val
        return -u_val - κ * laplacian
    
    # Adaptive mesh refinement loop
    # Try different resolutions and element degrees
    resolutions = [32, 64, 128, 256]
    element_degrees = [1, 2]  # Try P1 and P2 elements
    
    best_solution = None
    best_error = float('inf')
    best_params = None
    best_solver_info = None
    
    # Solver info template
    solver_info_template = {
        "mesh_resolution": None,
        "element_degree": None,
        "ksp_type": None,
        "pc_type": None,
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt_suggested,
        "n_steps": int(t_end / dt_suggested),
        "time_scheme": scheme,
    }
    
    # Time stepping parameters
    dt = dt_suggested
    n_steps = int(t_end / dt)
    if n_steps == 0:
        n_steps = 1
        dt = t_end
    
    # Try iterative solver first, fallback to direct
    solver_strategy = "iterative"
    
    for degree in element_degrees:
        for N in resolutions:
            if rank == 0:
                print(f"Testing N={N}, degree={degree}")
            
            # Check time constraint - if we're taking too long, break
            current_time = time.time()
            if current_time - start_time > 15.0:  # Leave some margin
                if rank == 0:
                    print("  Time limit approaching, stopping refinement")
                break
            
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Boundary condition: Dirichlet from exact solution
            tdim = domain.topology.dim
            fdim = tdim - 1
            
            # Define boundary marker for all boundaries
            def boundary_marker(x):
                return np.logical_or.reduce([
                    np.isclose(x[0], 0.0),
                    np.isclose(x[0], 1.0),
                    np.isclose(x[1], 0.0),
                    np.isclose(x[1], 1.0)
                ])
            
            boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            # Create BC function (will be updated at each time step)
            u_bc = fem.Function(V)
            
            # Initial condition
            u_n = fem.Function(V)
            u_n.interpolate(lambda x: u_exact(x, 0.0))
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            κ = fem.Constant(domain, ScalarType(1.0))
            
            # Time-stepping scheme: backward Euler
            a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(fem.Constant(domain, ScalarType(0.0)), v) * ufl.dx
            
            # Assemble forms
            a_form = fem.form(a)
            L_form = fem.form(L)
            
            # Assemble matrix (without BCs initially)
            A = petsc.assemble_matrix(a_form)
            A.assemble()
            
            # Create vectors
            b = petsc.create_vector(L_form.function_spaces)
            u_sol = fem.Function(V)
            
            # Configure linear solver
            solver = PETSc.KSP().create(domain.comm)
            
            if solver_strategy == "iterative":
                # Try iterative solver first
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
            else:
                # Fallback to direct solver
                solver.setType(PETSc.KSP.Type.PREONLY)
                solver.getPC().setType(PETSc.PC.Type.LU)
            
            solver.setOperators(A)
            solver.setFromOptions()
            
            # Time stepping loop
            total_iterations = 0
            t = 0.0
            solver_converged_all_steps = True
            
            for step in range(n_steps):
                t += dt
                
                # Update boundary condition with exact solution at current time
                u_bc.interpolate(lambda x: u_exact(x, t))
                bc = fem.dirichletbc(u_bc, dofs)
                
                # Create source term function for current time
                f_func = fem.Function(V)
                f_func.interpolate(lambda x: f_source(x, t))
                
                # Update RHS form with current source term
                L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
                L_form = fem.form(L)
                
                # Assemble RHS
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                
                # Apply BCs to matrix (zero rows, set diagonal to 1 for Dirichlet dofs)
                # First, get the indices of dofs for BCs
                bc_dofs = bc.dof_indices()[0]
                if len(bc_dofs) > 0:
                    # Zero the rows in the matrix
                    A.zeroRows(bc_dofs, diag=1.0)
                
                # Apply lifting and BCs to RHS
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Solve linear system
                solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                
                # Get iteration count
                total_iterations += solver.getIterationNumber()
                
                # Check if solver converged for this step
                if not solver.is_converged:
                    solver_converged_all_steps = False
                
                # Update u_n for next step
                u_n.x.array[:] = u_sol.x.array[:]
                
                # Reassemble matrix for next step (undo BC modifications)
                if step < n_steps - 1:
                    # Reassemble the matrix for next time step
                    A.zeroEntries()
                    petsc.assemble_matrix(A, a_form)
                    A.assemble()
            
            # Compute error against exact solution at final time
            # Create exact solution function
            u_exact_func = fem.Function(V)
            u_exact_func.interpolate(lambda x: u_exact(x, t_end))
            
            # Compute L2 error
            error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
            error_value = np.sqrt(fem.assemble_scalar(error_form))
            
            if rank == 0:
                print(f"  N={N}, degree={degree}: L2 error = {error_value:.2e}")
            
            # Keep track of best solution
            if error_value < best_error:
                best_error = error_value
                best_solution = (u_sol, domain, V)
                best_params = (N, degree)
                
                # Create solver info for this best solution
                solver_info = solver_info_template.copy()
                solver_info["mesh_resolution"] = N
                solver_info["element_degree"] = degree
                if solver_strategy == "iterative":
                    solver_info["ksp_type"] = "gmres"
                    solver_info["pc_type"] = "hypre"
                else:
                    solver_info["ksp_type"] = "preonly"
                    solver_info["pc_type"] = "lu"
                solver_info["iterations"] = total_iterations
                solver_info["dt"] = dt
                solver_info["n_steps"] = n_steps
                solver_info["time_scheme"] = scheme
                best_solver_info = solver_info
            
            # Check if error meets accuracy requirement
            if error_value <= 2.24e-04:
                if rank == 0:
                    print(f"  Accuracy requirement met at N={N}, degree={degree}")
                # Use this solution
                best_solution = (u_sol, domain, V)
                best_params = (N, degree)
                best_solver_info = solver_info
                break
            
            # If iterative solver failed, switch to direct for next resolution
            if solver_strategy == "iterative" and not solver_converged_all_steps:
                if rank == 0:
                    print("  Iterative solver failed, switching to direct solver")
                solver_strategy = "direct"
        
        # Break outer loop if accuracy requirement met
        if best_error <= 2.24e-04:
            break
    
    # Use the best solution found
    if best_solution is None:
        # Fallback: use last tried configuration
        best_solution = (u_sol, domain, V)
        best_params = (N, degree)
    
    u_final, domain_final, V_final = best_solution
    N_final, degree_final = best_params
    
    if rank == 0:
        print(f"Selected N={N_final}, degree={degree_final} with error {best_error:.2e}")
    
    # Create 50x50 uniform grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Prepare points for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
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
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Handle any remaining NaN values (points not found on any processor)
    # In serial, this shouldn't happen, but for safety:
    u_values = comm.allreduce(u_values, op=MPI.MAX)
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also get initial condition on the same grid
    u0_func = fem.Function(V_final)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_values = comm.allreduce(u0_values, op=MPI.MAX)
    u0_grid = u0_values.reshape((nx, ny))
    
    # Check time constraint
    end_time = time.time()
    elapsed = end_time - start_time
    if rank == 0:
        print(f"Total solver time: {elapsed:.3f}s")
        if elapsed > 16.584:
            print(f"WARNING: Time constraint (16.584s) may not be met: {elapsed:.3f}s > 16.584s")
        else:
            print(f"Time constraint satisfied: {elapsed:.3f}s <= 16.584s")
    
    # Return results
    result = {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": best_solver_info
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a minimal case specification
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.008,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Solution shape: {result['u'].shape}")

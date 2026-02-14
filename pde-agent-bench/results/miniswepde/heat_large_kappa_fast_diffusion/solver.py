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
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    pde_info = case_spec.get('pde', {})
    time_info = pde_info.get('time', {})
    
    # Time parameters - use provided values or defaults from problem description
    t_end = time_info.get('t_end', 0.08)
    dt_suggested = time_info.get('dt', 0.004)
    scheme = time_info.get('scheme', 'backward_euler')
    
    # Coefficients
    kappa = pde_info.get('coefficients', {}).get('kappa', 5.0)
    
    # Domain
    domain_info = pde_info.get('domain', {})
    if not domain_info:
        domain_info = {'type': 'unit_square', 'bounds': [[0, 1], [0, 1]]}
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with linear elements
    
    # Solver parameters
    ksp_type = 'gmres'
    pc_type = 'hypre'
    rtol = 1e-8
    
    # For tracking convergence
    prev_error = None
    converged_resolution = None
    final_solution = None
    final_mesh = None
    final_V = None
    final_error = None
    
    # Time stepping counters
    total_linear_iterations = 0
    n_steps_actual = 0
    dt_actual = dt_suggested
    
    # Function for exact solution evaluation
    def exact_sol(x, t):
        return np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing resolution N={N}")
        
        # Create mesh
        if domain_info.get('type') == 'unit_square':
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        else:
            bounds = domain_info.get('bounds', [[0, 1], [0, 1]])
            p0 = np.array([bounds[0][0], bounds[1][0]])
            p1 = np.array([bounds[0][1], bounds[1][1]])
            domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Time constant
        t_const = fem.Constant(domain, ScalarType(0.0))
        
        # BC function
        u_bc = fem.Function(V)
        
        def update_bc(t_val):
            t_const.value = t_val
            u_bc.interpolate(lambda x: exact_sol(x, t_val))
        
        update_bc(0.0)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Initial condition
        u_n = fem.Function(V)
        u_n.interpolate(lambda x: exact_sol(x, 0.0))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        # Time-stepping
        dt = dt_suggested
        t = 0.0
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps
        
        # Source term: f = (-1 + 5κπ²)*exp(-t)*sin(2πx)*sin(πy)
        f_expr = (-1.0 + 5.0*kappa*ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        
        # Backward Euler
        a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Setup linear solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        try:
            solver.setType(ksp_type)
            solver.getPC().setType(pc_type)
            solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
        except:
            solver.setType('preonly')
            solver.getPC().setType('lu')
        
        # Time stepping loop
        linear_iterations_this_resolution = 0
        for step in range(n_steps):
            t += dt
            
            # Update time
            t_const.value = t
            update_bc(t)
            
            # Recreate L form
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
            L_form = fem.form(L)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply lifting and BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            
            # Get iteration count
            linear_iterations_this_resolution += solver.getIterationNumber()
            
            # Update for next step
            u_n.x.array[:] = u_sol.x.array
        
        total_linear_iterations += linear_iterations_this_resolution
        n_steps_actual = n_steps
        dt_actual = dt
        
        # Compute exact solution at final time
        u_exact_final = fem.Function(V)
        u_exact_final.interpolate(lambda x: exact_sol(x, t_end))
        
        # Compute L2 error
        error_expr = ufl.inner(u_sol - u_exact_final, u_sol - u_exact_final) * ufl.dx
        error_form = fem.form(error_expr)
        error_sq = fem.assemble_scalar(error_form)
        error = np.sqrt(error_sq)
        
        if rank == 0:
            print(f"  L2 error: {error:.6e}")
            print(f"  Solution norm: {np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))):.6f}")
            print(f"  Exact norm: {np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_exact_final, u_exact_final) * ufl.dx))):.6f}")
        
        # Check convergence based on error
        if prev_error is not None:
            error_change = abs(error - prev_error) / error if error > 0 else 0
            if rank == 0:
                print(f"  Error change: {error_change:.6f}")
            
            if error_change < 0.5:  # Error is not decreasing much
                converged_resolution = N
                final_solution = u_sol
                final_mesh = domain
                final_V = V
                final_error = error
                if rank == 0:
                    print(f"  Converged at resolution N={N}")
                break
        
        prev_error = error
        final_solution = u_sol
        final_mesh = domain
        final_V = V
        final_error = error
    
    if converged_resolution is None:
        converged_resolution = resolutions[-1]
        if rank == 0:
            print(f"Using finest resolution N={converged_resolution}")
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if rank == 0:
        print(f"Total solve time: {elapsed_time:.3f} seconds")
        print(f"Final L2 error: {final_error:.6e}")
        if final_error <= 9.91e-04:
            print(f"✓ Accuracy requirement met: {final_error:.6e} <= 9.91e-04")
        else:
            print(f"✗ Accuracy requirement NOT met: {final_error:.6e} > 9.91e-04")
        
        if elapsed_time <= 26.034:
            print(f"✓ Time requirement met: {elapsed_time:.3f}s <= 26.034s")
        else:
            print(f"✗ Time requirement NOT met: {elapsed_time:.3f}s > 26.034s")
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    u_grid_flat = np.full((nx * ny,), np.nan)
    
    if final_solution is not None:
        bb_tree = geometry.bb_tree(final_mesh, final_mesh.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(final_mesh, cell_candidates, points.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        if len(points_on_proc) > 0:
            vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_grid_flat[eval_map] = vals.flatten()
    
    if comm.size > 1:
        u_grid_all = comm.gather(u_grid_flat, root=0)
        if rank == 0:
            u_grid_combined = np.full((nx * ny,), np.nan)
            for arr in u_grid_all:
                mask = ~np.isnan(arr)
                u_grid_combined[mask] = arr[mask]
            u_grid_flat = u_grid_combined
        else:
            u_grid_flat = np.full((nx * ny,), np.nan)
    
    if rank == 0:
        u_grid = u_grid_flat.reshape((nx, ny))
        
        u0_func = fem.Function(final_V)
        u0_func.interpolate(lambda x: exact_sol(x, 0.0))
        
        u0_grid_flat = np.full((nx * ny,), np.nan)
        points_on_proc0 = []
        cells_on_proc0 = []
        eval_map0 = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc0.append(points.T[i])
                cells_on_proc0.append(links[0])
                eval_map0.append(i)
        
        if len(points_on_proc0) > 0:
            vals0 = u0_func.eval(np.array(points_on_proc0), np.array(cells_on_proc0, dtype=np.int32))
            u0_grid_flat[eval_map0] = vals0.flatten()
        
        u0_grid = u0_grid_flat.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny))
        u0_grid = np.zeros((nx, ny))
    
    solver_info = {
        "mesh_resolution": converged_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_linear_iterations,
        "dt": dt_actual,
        "n_steps": n_steps_actual,
        "time_scheme": scheme
    }
    
    result = {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a sample case specification
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.08,
                "dt": 0.004,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": 5.0
            },
            "domain": {
                "type": "unit_square",
                "bounds": [[0, 1], [0, 1]]
            }
        }
    }
    
    result = solve(case_spec)
    print("\nSolver info:", result["solver_info"])
    print("Solution shape:", result["u"].shape)
    print("Initial condition shape:", result["u_initial"].shape)

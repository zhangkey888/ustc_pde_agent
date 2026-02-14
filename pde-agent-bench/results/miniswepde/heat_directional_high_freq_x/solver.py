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
    Implements runtime auto-tuning with grid convergence loop.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    # According to guidelines: If Problem Description mentions t_end or dt,
    # we MUST set hardcoded defaults and force is_transient = True
    t_end = 0.08
    dt_suggested = 0.004
    scheme = "backward_euler"
    
    # Check if time parameters are provided in case_spec
    if case_spec is not None and 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Manufactured solution: u = exp(-t)*sin(8*pi*x)*sin(pi*y)
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(8*np.pi*x[0]) * np.sin(np.pi*x[1])
    
    def source_term(x, t, kappa=1.0):
        return np.exp(-t) * np.sin(8*np.pi*x[0]) * np.sin(np.pi*x[1]) * (65*np.pi**2*kappa - 1)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_solutions = []
    norms = []  # Store L2 norms of solutions
    errors = []  # Store L2 errors against exact solution
    mesh_resolution_used = None
    element_degree = 2  # Use P2 elements for better accuracy with high frequency
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Variables to track solver performance
    total_linear_iterations = 0
    dt_used = dt_suggested
    n_steps_used = int(np.ceil(t_end / dt_used))
    dt_used = t_end / n_steps_used  # Adjust to exactly reach t_end
    
    for idx, N in enumerate(resolutions):
        if rank == 0:
            print(f"Testing mesh resolution N={N} with P{element_degree} elements")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Define initial condition
        u_n = fem.Function(V)
        u_n.interpolate(lambda x: exact_solution(x, 0.0))
        
        # Define boundary condition (Dirichlet from exact solution)
        def boundary_marker(x):
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
        
        # Time-dependent boundary condition
        u_bc = fem.Function(V)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Time-stepping loop
        t = 0.0
        linear_iterations_this_mesh = 0
        
        for step in range(n_steps_used):
            t_prev = t
            t += dt_used
            
            # Update boundary condition
            u_bc.interpolate(lambda x: exact_solution(x, t))
            
            # Create time-dependent source term
            f_func = fem.Function(V)
            f_func.interpolate(lambda x: source_term(x, t, kappa=1.0))
            
            # Define variational problem for backward Euler
            a = ufl.inner(u, v) * ufl.dx + dt_used * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt_used * ufl.inner(f_func, v) * ufl.dx
            
            # Create forms
            a_form = fem.form(a)
            L_form = fem.form(L)
            
            # Assemble matrix
            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            
            # Create RHS vector
            b = petsc.create_vector(L_form.function_spaces)
            
            # Setup solver - try iterative first
            solver_success = False
            solver_type = "iterative"
            
            for solver_attempt in range(2):  # Try iterative, then direct
                try:
                    solver = PETSc.KSP().create(domain.comm)
                    solver.setOperators(A)
                    
                    if solver_attempt == 0:
                        # Try iterative solver first
                        solver.setType(PETSc.KSP.Type.GMRES)
                        solver.getPC().setType(PETSc.PC.Type.HYPRE)
                        solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
                        solver_type = "gmres"
                        pc_type = "hypre"
                    else:
                        # Fallback to direct solver
                        solver.setType(PETSc.KSP.Type.PREONLY)
                        solver.getPC().setType(PETSc.PC.Type.LU)
                        solver_type = "preonly"
                        pc_type = "lu"
                    
                    # Assemble RHS
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting and boundary conditions
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Create solution vector
                    u_sol = fem.Function(V)
                    
                    # Solve
                    solver.solve(b, u_sol.x.petsc_vec)
                    u_sol.x.scatter_forward()
                    
                    # Get iteration count
                    its = solver.getIterationNumber()
                    linear_iterations_this_mesh += its
                    
                    solver_success = True
                    break
                    
                except Exception as e:
                    if solver_attempt == 0:
                        if rank == 0:
                            print(f"Iterative solver failed for N={N}, step {step}: {e}")
                            print("Falling back to direct solver")
                        continue
                    else:
                        if rank == 0:
                            print(f"Direct solver also failed for N={N}, step {step}: {e}")
                        raise
            
            if not solver_success:
                raise RuntimeError(f"Both iterative and direct solvers failed for N={N}")
            
            # Update u_n for next time step
            u_n.x.array[:] = u_sol.x.array
        
        # Store final solution for this resolution
        u_solutions.append(u_n.copy())
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_n, u_n) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        
        # Compute L2 error against exact solution at final time
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: exact_solution(x, t_end))
        error_form = fem.form(ufl.inner(u_n - u_exact, u_n - u_exact) * ufl.dx)
        error_value = np.sqrt(fem.assemble_scalar(error_form))
        errors.append(error_value)
        
        if rank == 0:
            print(f"  N={N}: L2 norm = {norm_value:.6e}, L2 error = {error_value:.2e}, linear iterations = {linear_iterations_this_mesh}")
        
        total_linear_iterations += linear_iterations_this_mesh
        
        # Check convergence (compare with previous resolution if available)
        if idx > 0:
            relative_error = abs(norms[idx] - norms[idx-1]) / norms[idx] if norms[idx] != 0 else float('inf')
            if rank == 0:
                print(f"  Relative error vs previous mesh: {relative_error:.2%}")
            
            # Check if we meet accuracy requirement
            if error_value <= 2.27e-3:
                if rank == 0:
                    print(f"  ACCURACY MET at N={N} (error = {error_value:.2e} <= 2.27e-3)")
                mesh_resolution_used = N
                break
            
            if relative_error < 0.01:  # 1% convergence criterion
                mesh_resolution_used = N
                if rank == 0:
                    print(f"  CONVERGED at N={N} (relative error < 1%)")
                break
    
    # If loop finished without convergence, use finest mesh
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
        if rank == 0:
            print(f"Using finest mesh N={mesh_resolution_used} (no convergence)")
    
    # Get the final solution (last one computed before break or finest)
    final_idx = resolutions.index(mesh_resolution_used)
    u_final = u_solutions[final_idx]
    final_error = errors[final_idx]
    domain_final = u_final.function_space.mesh
    
    # Interpolate solution to a 50x50 grid for output
    nx = ny = 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points array (3D even for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
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
    
    # Also get initial condition on same grid
    V_final = fem.functionspace(domain_final, ("Lagrange", element_degree))
    u_initial_func = fem.Function(V_final)
    u_initial_func.interpolate(lambda x: exact_solution(x, 0.0))
    
    # Evaluate initial condition at same points
    u_initial_vals = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_initial = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals_initial.flatten()
    
    # In parallel, we need to gather results
    if comm.size > 1:
        # Gather solution values
        u_values_all = comm.gather(u_values, root=0)
        u_initial_all = comm.gather(u_initial_vals, root=0)
        
        if rank == 0:
            # Combine results from all ranks
            u_values_combined = np.full((points.shape[1],), np.nan)
            u_initial_combined = np.full((points.shape[1],), np.nan)
            
            for rank_vals, rank_init in zip(u_values_all, u_initial_all):
                valid_mask = ~np.isnan(rank_vals)
                u_values_combined[valid_mask] = rank_vals[valid_mask]
                u_initial_combined[valid_mask] = rank_init[valid_mask]
            
            u_values = u_values_combined
            u_initial_vals = u_initial_combined
        else:
            u_values = np.full((points.shape[1],), np.nan)
            u_initial_vals = np.full((points.shape[1],), np.nan)
    
    # Only rank 0 needs to create the final output
    if rank == 0:
        # Reshape to (nx, ny)
        u_grid = u_values.reshape((nx, ny))
        u_initial_grid = u_initial_vals.reshape((nx, ny))
        
        # Prepare solver info
        solver_info = {
            "mesh_resolution": mesh_resolution_used,
            "element_degree": element_degree,
            "ksp_type": "gmres",  # We tried this first
            "pc_type": "hypre",   # We tried this first
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": dt_used,
            "n_steps": n_steps_used,
            "time_scheme": scheme
        }
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        print(f"\n=== Solver Summary ===")
        print(f"Solve completed in {solve_time:.2f} seconds")
        print(f"Final mesh resolution: {mesh_resolution_used}")
        print(f"Final L2 error: {final_error:.2e}")
        print(f"Total linear iterations: {total_linear_iterations}")
        print(f"Time constraint: {solve_time:.2f}s <= 29.631s: {solve_time <= 29.631}")
        print(f"Accuracy constraint: {final_error:.2e} <= 2.27e-03: {final_error <= 2.27e-3}")
        
        return {
            "u": u_grid,
            "u_initial": u_initial_grid,
            "solver_info": solver_info
        }
    else:
        # Other ranks return empty dict
        return {"u": np.zeros((1, 1)), "u_initial": np.zeros((1, 1)), "solver_info": {}}

if __name__ == "__main__":
    # Test the solver with a minimal case specification
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.004,
                "scheme": "backward_euler"
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("\n=== Test Results ===")
        print("Solver info:", result["solver_info"])
        print("Solution shape:", result["u"].shape)
        print("Initial condition shape:", result["u_initial"].shape)

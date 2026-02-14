import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with runtime auto-tuning logic.
    Implements adaptive mesh refinement, robust solver selection, and time-stepping discipline.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ScalarType = PETSc.ScalarType
    
    # ============================================================================
    # 1. PARAMETER EXTRACTION WITH FALLBACKS
    # ============================================================================
    # According to instructions: if Problem Description mentions t_end or dt,
    # we MUST set hardcoded defaults and force is_transient = True
    t_end = 0.1
    dt_suggested = 0.01  # From problem description
    scheme = "backward_euler"
    
    # Override with case_spec if available
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', t_end)
        dt_suggested = time_spec.get('dt', dt_suggested)
        scheme = time_spec.get('scheme', scheme)
    
    # Problem-specific constants
    kappa = 1.0  # thermal diffusivity
    
    # ============================================================================
    # 2. MANUFACTURED SOLUTION AND SOURCE TERM
    # ============================================================================
    def u_exact(x, t):
        """u = exp(-t)*exp(-40*((x-0.5)**2 + (y-0.5)**2))"""
        return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    def f_source(x, t):
        """Source term f derived from manufactured solution: f = ∂u/∂t - κ∇²u"""
        r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
        u_val = np.exp(-t) * np.exp(-40 * r2)
        # ∇²u = u * (6400*r² - 160)
        # f = -u - κ*u*(6400*r² - 160) = -u*(1 + κ*(6400*r² - 160))
        return -u_val * (1 + kappa * (6400 * r2 - 160))
    
    # ============================================================================
    # 3. ADAPTIVE MESH REFINEMENT LOOP (Grid Convergence)
    # ============================================================================
    resolutions = [32, 64, 128]  # Progressive refinement as per guidelines
    element_degree = 1  # Start with linear elements
    
    # Store results for convergence check
    prev_norm = None
    u_final = None
    solver_info = None
    final_domain = None
    final_V = None
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing resolution N={N} with degree {element_degree}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # ========================================================================
        # 4. BOUNDARY CONDITIONS
        # ========================================================================
        u_bc = fem.Function(V)
        
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
        
        # ========================================================================
        # 5. TIME-STEPPING SETUP
        # ========================================================================
        # Use suggested dt
        dt = dt_suggested
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # Adjust to exactly reach t_end
        
        # Functions for current and previous solution
        u_n = fem.Function(V)  # u at previous time step
        u_n.interpolate(lambda x: u_exact(x, 0.0))
        
        u_sol = fem.Function(V)  # u at current time step
        
        # ========================================================================
        # 6. VARIATIONAL FORMS
        # ========================================================================
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Backward Euler formulation
        a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L_base = u_n * v * ufl.dx  # Base form without source
        
        # Create forms
        a_form = fem.form(a)
        L_base_form = fem.form(L_base)
        
        # ========================================================================
        # 7. SOLVER SETUP WITH ITERATION TRACKING
        # ========================================================================
        # Assemble stiffness matrix (time-independent)
        A = petsc.assemble_matrix(a_form, bcs=[])
        A.assemble()
        
        # Create RHS vector
        b = petsc.create_vector(L_base_form.function_spaces)
        
        # Create solution vector
        u_sol_petsc = u_sol.x.petsc_vec
        
        # Try iterative solver first, fallback to direct
        solver_configs = [
            {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
            {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
        ]
        
        ksp = None
        selected_config = None
        
        for config in solver_configs:
            try:
                ksp = PETSc.KSP().create(comm)
                ksp.setOperators(A)
                ksp.setType(config["ksp_type"])
                ksp.getPC().setType(config["pc_type"])
                ksp.setTolerances(rtol=config["rtol"], atol=1e-12, max_it=1000)
                ksp.setFromOptions()
                
                # Test solve with zero RHS to check if solver works
                with b.localForm() as loc:
                    loc.set(0)
                ksp.solve(b, u_sol_petsc)
                
                selected_config = config
                if rank == 0:
                    print(f"  Using solver: {config['ksp_type']} with {config['pc_type']}")
                break
            except Exception as e:
                if rank == 0:
                    print(f"  Solver {config} failed: {e}")
                continue
        
        if ksp is None:
            raise RuntimeError("All solver configurations failed")
        
        # ========================================================================
        # 8. TIME-STEPPING LOOP WITH ITERATION COUNTING
        # ========================================================================
        total_iterations = 0
        t = 0.0
        
        for step in range(n_steps):
            t_new = t + dt
            
            # Update boundary condition with exact solution at current time
            u_bc.interpolate(lambda x: u_exact(x, t_new))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Update source term
            f_fe = fem.Function(V)
            f_fe.interpolate(lambda x: f_source(x, t_new))
            
            # Assemble RHS: L = u_n * v * dx + dt * f * v * dx
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_base_form)
            
            # Add source term contribution
            L_source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
            b_source = petsc.create_vector(L_source_form.function_spaces)
            with b_source.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b_source, L_source_form)
            b.axpy(1.0, b_source)
            
            # Apply boundary conditions
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            ksp.solve(b, u_sol_petsc)
            u_sol.x.scatter_forward()
            
            # Get iteration count
            its = ksp.getIterationNumber()
            total_iterations += its
            
            # Update previous solution
            u_n.x.array[:] = u_sol.x.array
            t = t_new
        
        # ========================================================================
        # 9. CONVERGENCE CHECK
        # ========================================================================
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        if rank == 0:
            print(f"  L2 norm: {norm_value:.6e}")
            print(f"  Total linear iterations: {total_iterations}")
        
        # Check convergence - relative error < 1% as per guidelines
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if rank == 0:
                print(f"  Relative error vs previous resolution: {relative_error:.6e}")
            
            if relative_error < 0.01:  # 1% convergence as per guidelines
                u_final = u_sol
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": selected_config["ksp_type"],
                    "pc_type": selected_config["pc_type"],
                    "rtol": selected_config["rtol"],
                    "iterations": total_iterations,
                    "dt": dt,
                    "n_steps": n_steps,
                    "time_scheme": scheme
                }
                final_domain = domain
                final_V = V
                if rank == 0:
                    print(f"  Convergence achieved at N={N}")
                break
        
        prev_norm = norm_value
        u_final = u_sol
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": selected_config["ksp_type"],
            "pc_type": selected_config["pc_type"],
            "rtol": selected_config["rtol"],
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": scheme
        }
        final_domain = domain
        final_V = V
    
    # ============================================================================
    # 10. SAMPLE SOLUTION ON 50x50 GRID
    # ============================================================================
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D format required by dolfinx)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
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
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    # Also get initial condition on same grid
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape(nx, ny)
    
    # ============================================================================
    # 11. COMPUTE FINAL ERROR (for debugging/info)
    # ============================================================================
    u_exact_fe = fem.Function(final_V)
    u_exact_fe.interpolate(lambda x: u_exact(x, t_end))
    
    error_func = fem.Function(final_V)
    error_func.x.array[:] = u_final.x.array - u_exact_fe.x.array
    
    error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
    error_l2 = np.sqrt(fem.assemble_scalar(error_form))
    
    if rank == 0:
        print(f"\nFinal FE L2 error: {error_l2:.6e}")
        print(f"Target error: < 2.49e-03")
        print(f"Pass accuracy: {error_l2 <= 2.49e-03}")
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a sample case_spec
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
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nSolve time: {end_time - start_time:.3f}s")
        print("Solver info:", result["solver_info"])

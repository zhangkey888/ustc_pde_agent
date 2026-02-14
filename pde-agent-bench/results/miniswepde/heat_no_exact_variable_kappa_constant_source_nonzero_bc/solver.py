import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx import nls
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing PDE specification with keys:
        - 'pde': dict with 'time' key containing 't_end', 'dt', 'scheme'
        - 'coefficients': dict with 'kappa' expression
        - 'source': float or expression
        - 'initial_condition': float or expression
        - 'boundary_condition': dict with 'type' and 'value'
    
    Returns:
    --------
    dict with keys:
        - 'u': numpy array of shape (50, 50) with solution values
        - 'u_initial': numpy array of shape (50, 50) with initial condition
        - 'solver_info': dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters with defaults
    pde_info = case_spec.get('pde', {})
    time_info = pde_info.get('time', {})
    
    # Time parameters - use defaults from problem description
    t_end = time_info.get('t_end', 0.1)
    dt_suggested = time_info.get('dt', 0.02)
    scheme = time_info.get('scheme', 'backward_euler')
    
    # Source term
    f_value = case_spec.get('source', 1.0)
    
    # Initial condition
    u0_value = case_spec.get('initial_condition', 0.0)
    
    # Coefficients
    coeffs = case_spec.get('coefficients', {})
    kappa_info = coeffs.get('kappa', {})
    kappa_expr_str = kappa_info.get('expr', '1.0')
    
    # Domain is unit square [0,1]x[0,1]
    domain_bounds = [[0.0, 0.0], [1.0, 1.0]]
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    
    # For storing convergence data
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_u = None
    final_u_initial = None
    final_domain = None
    
    # Solver info to collect
    total_linear_iterations = 0
    n_steps_actual = 0
    dt_actual = dt_suggested
    
    # Time-stepping loop control
    max_retries = 3
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_rectangle(
            comm, 
            domain_bounds, 
            [N, N], 
            cell_type=mesh.CellType.triangle
        )
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define kappa as expression
        x = ufl.SpatialCoordinate(domain)
        # Parse kappa expression string safely
        namespace = {
            'sin': ufl.sin, 'cos': ufl.cos, 'pi': np.pi, 
            'exp': ufl.exp, 'x': x[0], 'y': x[1],
            'np': np, 'np.sin': ufl.sin, 'np.cos': ufl.cos,
            'np.pi': np.pi, 'np.exp': ufl.exp
        }
        try:
            kappa_expr = eval(kappa_expr_str, namespace)
        except Exception as e:
            if rank == 0:
                print(f"  Warning: Could not parse kappa expression '{kappa_expr_str}': {e}")
                print(f"  Using constant kappa=1.0 instead")
            kappa_expr = 1.0
        
        # Define source term
        f = fem.Constant(domain, PETSc.ScalarType(f_value))
        
        # Define initial condition
        u0 = fem.Function(V)
        u0.interpolate(lambda x: np.full_like(x[0], u0_value))
        
        # Boundary condition (Dirichlet, u=0 on all boundaries)
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
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.full_like(x[0], 0.0))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Solution at previous time step
        u_n.x.array[:] = u0.x.array[:]
        
        # Define forms for time-dependent problem
        dt = fem.Constant(domain, PETSc.ScalarType(dt_actual))
        
        # Backward Euler discretization
        F = (1.0/dt) * ufl.inner(u - u_n, v) * ufl.dx + \
            ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx - \
            ufl.inner(f, v) * ufl.dx
        
        a = ufl.lhs(F)
        L = ufl.rhs(F)
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble stiffness matrix (time-independent part)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Setup linear solver with adaptive strategy
        # Try iterative solver first, fallback to direct if needed
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        for solver_attempt in range(2):  # Two attempts: iterative then direct
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            
            if solver_attempt == 0:
                # Try iterative solver
                ksp.setType(PETSc.KSP.Type.GMRES)
                ksp.getPC().setType(PETSc.PC.Type.HYPRE)
                ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                ksp_type = 'gmres'
                pc_type = 'hypre'
            else:
                # Fallback to direct solver
                ksp.setType(PETSc.KSP.Type.PREONLY)
                ksp.getPC().setType(PETSc.PC.Type.LU)
                ksp_type = 'preonly'
                pc_type = 'lu'
            
            ksp.setFromOptions()
            
            # Time-stepping loop
            t = 0.0
            n_steps = int(np.ceil(t_end / dt_actual))
            linear_iterations = 0
            step_success = True
            
            for step in range(n_steps):
                t += dt_actual
                if t > t_end + 1e-12:
                    break
                
                # Update u_n in the form (already done via u_n function)
                # Assemble RHS
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                
                # Apply lifting and boundary conditions
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Solve linear system
                try:
                    ksp.solve(b, u_sol.x.petsc_vec)
                    its = ksp.getIterationNumber()
                    linear_iterations += its
                    
                    if ksp.getConvergedReason() <= 0:
                        if solver_attempt == 0:
                            # Iterative solver failed, break and try direct
                            step_success = False
                            break
                        else:
                            # Direct solver failed, this is serious
                            raise RuntimeError("Direct solver failed to converge")
                    
                except Exception as e:
                    if solver_attempt == 0:
                        step_success = False
                        break
                    else:
                        raise
                
                # Update u_n for next step
                u_sol.x.scatter_forward()
                u_n.x.array[:] = u_sol.x.array[:]
            
            if step_success:
                solver_success = True
                total_linear_iterations += linear_iterations
                n_steps_actual = n_steps
                break
            else:
                # Clean up and try direct solver
                ksp.destroy()
                if rank == 0:
                    print(f"  Iterative solver failed, switching to direct solver")
        
        if not solver_success:
            if rank == 0:
                print(f"  Solver failed for N={N}, trying next resolution")
            continue
        
        # Compute norm for convergence check
        norm = np.sqrt(comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)), 
            op=MPI.SUM
        ))
        
        if rank == 0:
            print(f"  N={N}, norm={norm:.6e}, linear iterations={linear_iterations}")
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 0 else abs(norm - prev_norm)
            if rank == 0:
                print(f"  Relative error: {relative_error:.6e}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                final_solution = u_sol
                final_mesh_resolution = N
                final_u = u_sol
                final_u_initial = u0
                final_domain = domain
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
        
        prev_norm = norm
        final_solution = u_sol
        final_mesh_resolution = N
        final_u = u_sol
        final_u_initial = u0
        final_domain = domain
    
    # If loop finished without convergence, use finest mesh result
    if final_solution is None and len(resolutions) > 0:
        final_mesh_resolution = resolutions[-1]
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D even for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_grid = np.zeros((nx, ny))
    u_initial_grid = np.zeros((nx, ny))
    
    if final_u is not None and final_domain is not None:
        # Use geometry utilities for point evaluation
        bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(
            final_domain, cell_candidates, points.T
        )
        
        # Build per-point mapping
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
            # Evaluate final solution
            vals = final_u.eval(
                np.array(points_on_proc), 
                np.array(cells_on_proc, dtype=np.int32)
            )
            u_flat = np.full(points.shape[1], np.nan)
            u_flat[eval_map] = vals.flatten()
            u_grid = u_flat.reshape((nx, ny))
            
            # Evaluate initial condition
            vals_init = final_u_initial.eval(
                np.array(points_on_proc),
                np.array(cells_on_proc, dtype=np.int32)
            )
            u_init_flat = np.full(points.shape[1], np.nan)
            u_init_flat[eval_map] = vals_init.flatten()
            u_initial_grid = u_init_flat.reshape((nx, ny))
    
    # Fill NaN values with 0 (for points outside domain, though all should be inside)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    u_initial_grid = np.nan_to_num(u_initial_grid, nan=0.0)
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": final_mesh_resolution if final_mesh_resolution else resolutions[-1],
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_linear_iterations,
        "dt": dt_actual,
        "n_steps": n_steps_actual,
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

# Test the solver if run directly
if __name__ == "__main__":
    # Create a test case specification
    test_case = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "source": 1.0,
        "initial_condition": 0.0,
        "coefficients": {
            "kappa": {
                "type": "expr",
                "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"
            }
        }
    }
    
    print("Testing solver with heat equation...")
    start_time = time.time()
    result = solve(test_case)
    end_time = time.time()
    
    print(f"Solve completed in {end_time - start_time:.2f} seconds")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Time steps: {result['solver_info']['n_steps']}")
    print(f"Total linear iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

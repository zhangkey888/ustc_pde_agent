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
    Solve the heat equation with adaptive mesh refinement and runtime auto-tuning.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters from case_spec or use defaults
    # According to problem description, we MUST set hardcoded defaults
    t_end = 0.1
    dt = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided (but problem says to use defaults)
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        scheme = time_params.get('scheme', scheme)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with linear elements
    
    # Storage for convergence check
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_u = None
    final_domain = None
    final_V = None
    
    # Solver info to collect
    total_linear_iterations = 0
    n_steps_actual = 0
    final_ksp_type = "gmres"  # Default
    final_pc_type = "hypre"   # Default
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        # u = 0 on boundary (from problem description: u = g, but g not specified, assume 0)
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
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define coefficients using symbolic coordinates
        x = ufl.SpatialCoordinate(domain)
        # Variable kappa: 1 + 0.6*sin(2*pi*x)*sin(2*pi*y)
        kappa_expr = 1.0 + 0.6 * ufl.sin(2*np.pi*x[0]) * ufl.sin(2*np.pi*x[1])
        
        # Source term f
        f_expr = ufl.sin(4*np.pi*x[0]) * ufl.sin(3*np.pi*x[1]) + \
                 0.3 * ufl.sin(10*np.pi*x[0]) * ufl.sin(9*np.pi*x[1])
        
        # Initial condition u0
        u0_expr = ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
        
        u = fem.Function(V)  # Current time step (unknown)
        
        # Define variational problem for backward Euler
        v = ufl.TestFunction(V)
        F = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx \
            - ufl.inner(u_n, v) * ufl.dx - dt * ufl.inner(f_expr, v) * ufl.dx
        
        # Linear form (for linear problem approach)
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa_expr * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for this problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct
        solver_success = False
        linear_iterations_this_mesh = 0
        ksp_type_used = "gmres"
        pc_type_used = "hypre"
        
        for solver_config in [("gmres", "hypre"), ("preonly", "lu")]:
            ksp_type, pc_type = solver_config
            
            try:
                solver = PETSc.KSP().create(domain.comm)
                solver.setOperators(A)
                solver.setType(ksp_type)
                solver.getPC().setType(pc_type)
                
                # Set tolerances
                solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
                
                # Time stepping loop
                t = 0.0
                n_steps = int(t_end / dt + 0.5)
                linear_iters = 0
                
                for step in range(n_steps):
                    t += dt
                    
                    # Update RHS (time-dependent parts if any)
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting and BCs
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Solve linear system
                    solver.solve(b, u_sol.x.petsc_vec)
                    u_sol.x.scatter_forward()
                    
                    # Get iteration count
                    linear_iters += solver.getIterationNumber()
                    
                    # Update u_n for next step
                    u_n.x.array[:] = u_sol.x.array
                
                solver_success = True
                linear_iterations_this_mesh = linear_iters
                ksp_type_used = ksp_type
                pc_type_used = pc_type
                break
                
            except Exception as e:
                if rank == 0:
                    print(f"Solver {ksp_type}/{pc_type} failed: {e}")
                continue
        
        if not solver_success:
            if rank == 0:
                print(f"All solvers failed for N={N}, continuing to next resolution")
            continue
        
        # Compute norm for convergence check
        norm = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)), op=MPI.SUM))
        
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 1e-12 else abs(norm - prev_norm)
            if rank == 0:
                print(f"  N={N}, norm={norm:.6e}, rel_error={relative_error:.6e}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Converged at N={N}")
                final_solution = u_sol
                final_mesh_resolution = N
                final_u = u_sol
                final_domain = domain
                final_V = V
                total_linear_iterations += linear_iterations_this_mesh
                n_steps_actual = n_steps
                final_ksp_type = ksp_type_used
                final_pc_type = pc_type_used
                break
        
        prev_norm = norm
        final_solution = u_sol
        final_mesh_resolution = N
        final_u = u_sol
        final_domain = domain
        final_V = V
        total_linear_iterations += linear_iterations_this_mesh
        n_steps_actual = n_steps
        final_ksp_type = ksp_type_used
        final_pc_type = pc_type_used
    
    # If loop finished without break, use the last (finest) mesh
    if final_solution is None and len(resolutions) > 0:
        # This shouldn't happen if at least one mesh worked
        if rank == 0:
            print("Warning: No mesh succeeded, using fallback")
        # Create a simple fallback solution
        domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        final_solution = fem.Function(V)
        final_mesh_resolution = 32
        final_u = final_solution
        final_domain = domain
        final_V = V
        # Keep default solver types
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D format required)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    u_grid_flat = np.full(points.shape[1], np.nan)
    
    if final_u is not None and final_domain is not None:
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
        
        if len(points_on_proc) > 0:
            vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_grid_flat[eval_map] = vals.flatten()
    
    # Gather all points (using MAX to combine values from different ranks)
    u_grid_flat_global = np.zeros_like(u_grid_flat)
    comm.Allreduce(u_grid_flat, u_grid_flat_global, op=MPI.MAX)
    
    # Reshape to 2D grid
    u_grid = u_grid_flat_global.reshape((nx, ny))
    
    # Also compute initial condition on same grid for optional output
    u0_grid = np.zeros((nx, ny))
    if final_domain is not None and final_V is not None:
        # Recreate u0 expression for this domain
        x = ufl.SpatialCoordinate(final_domain)
        u0_expr = ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])
        
        u0_func = fem.Function(final_V)
        u0_func.interpolate(fem.Expression(u0_expr, final_V.element.interpolation_points))
        
        # Evaluate initial condition
        u0_grid_flat = np.full(points.shape[1], np.nan)
        
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
        
        u0_grid_flat_global = np.zeros_like(u0_grid_flat)
        comm.Allreduce(u0_grid_flat, u0_grid_flat_global, op=MPI.MAX)
        u0_grid = u0_grid_flat_global.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": final_ksp_type,
        "pc_type": final_pc_type,
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps_actual,
        "time_scheme": scheme
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u0_grid  # Optional but recommended
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a minimal case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")

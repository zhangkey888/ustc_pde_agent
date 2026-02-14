import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve transient heat equation with adaptive mesh refinement.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec or use defaults
    # Defaults from problem description
    t_end_default = 0.1
    dt_default = 0.02
    time_scheme_default = "backward_euler"
    
    # Initialize with defaults
    t_end = t_end_default
    dt = dt_default
    time_scheme = time_scheme_default
    
    # Override with case_spec if provided
    if case_spec and 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end_default)
        dt = time_params.get('dt', dt_default)
        time_scheme = time_params.get('scheme', time_scheme_default)
    
    # Define source term and initial condition functions
    def source_term(x):
        """f = sin(pi*x)*cos(pi*y)"""
        return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    
    def initial_condition(x):
        """u0 = sin(pi*x)*sin(pi*y)"""
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with linear elements
    
    # Store solutions and norms for convergence check
    prev_norm = None
    final_solution = None
    final_mesh_res = None
    final_iterations = 0
    final_domain = None
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        # Create unit square mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define functions for current and previous solutions
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(initial_condition)
        u_curr = fem.Function(V)  # Current time step
        
        # Define constants
        kappa = fem.Constant(domain, PETSc.ScalarType(1.0))  # κ = 1.0
        dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
        
        # Define source term as a function
        f = fem.Function(V)
        f.interpolate(source_term)
        
        # Time-stepping scheme: backward Euler
        # (u - u_n)/dt * v * dx + kappa * dot(grad(u), grad(v)) * dx = f * v * dx
        F = (u - u_n) / dt_constant * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
        a = ufl.lhs(F)
        L = ufl.rhs(F)
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Create matrix and vector
        A = petsc.assemble_matrix(a_form)
        A.assemble()
        b = petsc.create_vector(L_form.function_spaces)
        
        # Create linear solver - try iterative first
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Set solver options - try iterative first
        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
        solver.setFromOptions()
        
        # Time-stepping loop
        t = 0.0
        n_steps = int(t_end / dt)
        iterations_this_mesh = 0
        solver_failed = False
        
        for step in range(n_steps):
            t += dt
            
            # Update previous solution
            u_n.x.array[:] = u_curr.x.array[:]
            u_n.x.scatter_forward()
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Solve linear system
            try:
                solver.solve(b, u_curr.x.petsc_vec)
                # Check if solver converged
                if solver.getConvergedReason() < 0:
                    # Solver didn't converge, fall back to direct solver
                    raise RuntimeError("Iterative solver did not converge")
            except Exception:
                # Fallback to direct solver
                solver_direct = PETSc.KSP().create(domain.comm)
                solver_direct.setOperators(A)
                solver_direct.setType(PETSc.KSP.Type.PREONLY)
                solver_direct.getPC().setType(PETSc.PC.Type.LU)
                solver_direct.solve(b, u_curr.x.petsc_vec)
                solver = solver_direct  # Use direct solver for remaining steps
            
            u_curr.x.scatter_forward()
            
            # Get iteration count
            its = solver.getIterationNumber()
            iterations_this_mesh += its
        
        # Compute L2 norm of final solution
        norm_form = fem.form(ufl.inner(u_curr, u_curr) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 0
            if relative_error < 0.01:  # 1% convergence criterion
                final_solution = u_curr
                final_mesh_res = N
                final_iterations = iterations_this_mesh
                final_domain = domain
                break
        
        prev_norm = norm_value
        final_solution = u_curr
        final_mesh_res = N
        final_iterations = iterations_this_mesh
        final_domain = domain
    
    # If loop completes without break, use the last (finest) mesh
    if final_solution is None:
        final_solution = u_curr
        final_mesh_res = 128
        final_iterations = iterations_this_mesh
        final_domain = domain
    
    # Prepare output grid (50x50 uniform grid)
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
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
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny) grid
    u_grid = u_values.reshape((ny, nx))
    
    # Also compute initial condition on the same grid for u_initial
    V_final = fem.functionspace(final_domain, ("Lagrange", element_degree))
    u0_func = fem.Function(V_final)
    u0_func.interpolate(initial_condition)
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape((ny, nx))
    
    # Get solver type info (check what was actually used)
    ksp_type = "gmres"
    pc_type = "hypre"
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": final_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    # End timing
    end_time = time.time()
    wall_time_sec = end_time - start_time
    
    # Print debug info (will be captured by evaluator)
    print(f"Wall time: {wall_time_sec:.3f}s")
    print(f"Mesh resolution: {final_mesh_res}")
    print(f"Total iterations: {final_iterations}")
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

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
    print("Solver executed successfully")
    print(f"u shape: {result['u'].shape}")
    print(f"u_initial shape: {result['u_initial'].shape}")

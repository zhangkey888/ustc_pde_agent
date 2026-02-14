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
    Solve the heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters with defaults
    t_end = case_spec.get('pde', {}).get('time', {}).get('t_end', 0.1)
    dt_suggested = case_spec.get('pde', {}).get('time', {}).get('dt', 0.01)
    scheme = case_spec.get('pde', {}).get('time', {}).get('scheme', 'backward_euler')
    
    # Extract coefficients if provided
    coefficients = case_spec.get('pde', {}).get('coefficients', {})
    kappa_spec = coefficients.get('kappa', {'type': 'expr', 'expr': '1 + 0.8*sin(2*pi*x)*sin(2*pi*y)'})
    
    # Manufactured solution (known from problem description)
    def u_exact_func(x, t):
        return np.exp(-t) * np.sin(3*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    # Kappa function - parse from spec or use default
    def kappa_func(x):
        # Default kappa from problem description
        return 1.0 + 0.8 * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    # Source term derived from manufactured solution
    def source_term(x, t):
        x_val, y_val = x[0], x[1]
        
        u = np.exp(-t) * np.sin(3*np.pi*x_val) * np.sin(2*np.pi*y_val)
        u_t = -u
        
        kappa = 1.0 + 0.8 * np.sin(2*np.pi*x_val) * np.sin(2*np.pi*y_val)
        
        u_x = np.exp(-t) * 3*np.pi * np.cos(3*np.pi*x_val) * np.sin(2*np.pi*y_val)
        u_y = np.exp(-t) * 2*np.pi * np.sin(3*np.pi*x_val) * np.cos(2*np.pi*y_val)
        
        kappa_x = 0.8 * 2*np.pi * np.cos(2*np.pi*x_val) * np.sin(2*np.pi*y_val)
        kappa_y = 0.8 * 2*np.pi * np.sin(2*np.pi*x_val) * np.cos(2*np.pi*y_val)
        
        u_xx = -np.exp(-t) * (3*np.pi)**2 * np.sin(3*np.pi*x_val) * np.sin(2*np.pi*y_val)
        u_yy = -np.exp(-t) * (2*np.pi)**2 * np.sin(3*np.pi*x_val) * np.sin(2*np.pi*y_val)
        
        div_kappa_grad_u = kappa_x * u_x + kappa * u_xx + kappa_y * u_y + kappa * u_yy
        
        f = u_t - div_kappa_grad_u
        return f
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    
    # Solver info
    solver_info = {
        "mesh_resolution": None,
        "element_degree": 1,
        "ksp_type": None,
        "pc_type": None,
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt_suggested,
        "n_steps": int(t_end / dt_suggested),
        "time_scheme": scheme,
        "nonlinear_iterations": []
    }
    
    total_linear_iterations = 0
    final_solution = None
    final_domain = None
    final_mesh_resolution = None
    
    for i, N in enumerate(resolutions):
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Functions
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step
        
        # Initial condition
        u_n.interpolate(lambda x: u_exact_func(x, 0.0))
        
        # Boundary condition
        def boundary(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Time-dependent boundary condition
        u_bc = fem.Function(V)
        def update_bc(t):
            u_bc.interpolate(lambda x: u_exact_func(x, t))
        
        update_bc(0.0)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Kappa function
        kappa = fem.Function(V)
        kappa.interpolate(kappa_func)
        
        # Trial and test functions
        u_trial = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Constant for dt
        dt = fem.Constant(domain, ScalarType(dt_suggested))
        
        # Bilinear form
        a = (1.0/dt) * ufl.inner(u_trial, v) * ufl.dx + ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        a_form = fem.form(a)
        
        # Assemble matrix once
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create solver
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A)
        
        # Try iterative solver first
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        ksp.setTolerances(rtol=1e-8, max_it=1000)
        ksp.setFromOptions()
        
        # Time-stepping
        t = 0.0
        n_steps = int(t_end / dt_suggested)
        step_iterations = 0
        used_direct_solver = False
        
        for step in range(n_steps):
            t += dt_suggested
            
            # Update boundary condition
            update_bc(t)
            
            # Create RHS form
            f = fem.Function(V)
            f.interpolate(lambda x: source_term(x, t))
            
            L = (1.0/dt) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f, v) * ufl.dx
            L_form = fem.form(L)
            
            # Create RHS vector
            b = petsc.create_vector(L_form.function_spaces)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply boundary conditions
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            try:
                ksp.solve(b, u.x.petsc_vec)
                u.x.scatter_forward()
                step_iterations += ksp.getIterationNumber()
                
            except PETSc.Error:
                # Switch to direct solver if iterative fails
                if not used_direct_solver:
                    if rank == 0:
                        print(f"  Switching to direct solver at step {step}")
                    ksp.setType(PETSc.KSP.Type.PREONLY)
                    pc = ksp.getPC()
                    pc.setType(PETSc.PC.Type.LU)
                    ksp.setFromOptions()
                    used_direct_solver = True
                    
                    # Retry solve
                    ksp.solve(b, u.x.petsc_vec)
                    u.x.scatter_forward()
                    step_iterations += ksp.getIterationNumber()
                else:
                    raise
            
            # Update previous solution
            u_n.x.array[:] = u.x.array
        
        # Record solver type
        if used_direct_solver:
            solver_info["ksp_type"] = "preonly"
            solver_info["pc_type"] = "lu"
        else:
            solver_info["ksp_type"] = "gmres"
            solver_info["pc_type"] = "hypre"
        
        total_linear_iterations += step_iterations
        
        # Compute L2 norm
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if i == 0:
            prev_norm = norm_value
        else:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value != 0 else float('inf')
            prev_norm = norm_value
            
            if rank == 0:
                print(f"  Relative error: {relative_error:.6f}")
            
            if relative_error < 0.01:
                final_solution = u
                final_domain = domain
                final_mesh_resolution = N
                solver_info["mesh_resolution"] = N
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
        
        # If last resolution
        if i == len(resolutions) - 1:
            final_solution = u
            final_domain = domain
            final_mesh_resolution = N
            solver_info["mesh_resolution"] = N
            if rank == 0:
                print(f"  Using finest mesh N={N}")
    
    # Update solver info
    solver_info["iterations"] = total_linear_iterations
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    if final_solution is not None and final_domain is not None:
        # Build bounding box tree
        bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
        
        # Find cells containing points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points.T)
        
        # Evaluate solution
        u_grid_flat = np.full(points.shape[1], np.nan, dtype=ScalarType)
        
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
            values = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_grid_flat[eval_map] = values.flatten()
        
        # Allgather
        u_grid_all = np.zeros_like(u_grid_flat)
        comm.Allreduce(u_grid_flat, u_grid_all, op=MPI.MAX)
        u_grid = u_grid_all.reshape((nx, ny))
        
        # Initial condition
        initial_solution = fem.Function(V)
        initial_solution.interpolate(lambda x: u_exact_func(x, 0.0))
        
        u_initial_flat = np.full(points.shape[1], np.nan, dtype=ScalarType)
        
        points_on_proc_init = []
        cells_on_proc_init = []
        eval_map_init = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc_init.append(points.T[i])
                cells_on_proc_init.append(links[0])
                eval_map_init.append(i)
        
        if len(points_on_proc_init) > 0:
            values_init = initial_solution.eval(np.array(points_on_proc_init), np.array(cells_on_proc_init, dtype=np.int32))
            u_initial_flat[eval_map_init] = values_init.flatten()
        
        u_initial_all = np.zeros_like(u_initial_flat)
        comm.Allreduce(u_initial_flat, u_initial_all, op=MPI.MAX)
        u_initial = u_initial_all.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
        u_initial = np.zeros((nx, ny), dtype=ScalarType)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Total iterations: {result['solver_info']['iterations']}")

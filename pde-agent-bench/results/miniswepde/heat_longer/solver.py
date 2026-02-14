import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Time parameters from problem description (hardcoded as per instructions)
    t_end = 0.2
    dt = 0.02
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_info = case_spec['pde']['time']
        t_end = time_info.get('t_end', t_end)
        dt = time_info.get('dt', dt)
        time_scheme = time_info.get('scheme', time_scheme)
    
    # Coefficients
    kappa = 0.5
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_sol_final = None
    norm_old = None
    mesh_resolution_used = None
    element_degree = 1  # Use linear elements for efficiency
    
    for N in resolutions:
        if rank == 0:
            print(f"Trying mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition: Dirichlet on entire boundary
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
        
        # Create a dummy DirichletBC for matrix assembly (value doesn't matter)
        u_bc_dummy = fem.Function(V)
        bc_dummy = fem.dirichletbc(u_bc_dummy, dofs)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step
        f_func = fem.Function(V)  # Source term at current time
        u_bc = fem.Function(V)    # Boundary condition function
        
        # Initial condition at t=0
        u_n.interpolate(lambda x: np.exp(0) * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
        u.x.array[:] = u_n.x.array
        
        # Time loop
        n_steps = int(t_end / dt + 0.5)
        total_linear_iterations = 0
        
        # Define variational form for backward Euler
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        
        # Bilinear form (time-independent)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        a_form = fem.form(a)
        
        # Assemble matrix once (time-independent) with Dirichlet BC
        A = petsc.assemble_matrix(a_form, bcs=[bc_dummy])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector([V])
        u_sol_vec = u.x.petsc_vec
        
        # Create linear solver (direct for robustness)
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=1000)
        ksp.setFromOptions()
        
        # Time stepping
        for step in range(n_steps):
            current_time = (step + 1) * dt
            
            # Update boundary condition
            u_bc.interpolate(lambda x: np.exp(-2.0 * current_time) * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Update source term f
            f_val = (-2.0 + 2.0 * kappa * np.pi**2) * np.exp(-2.0 * current_time)
            f_func.interpolate(lambda x: f_val * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
            
            # Linear form
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
            L_form = fem.form(L)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            # Apply lifting for non-zero Dirichlet BC
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # Set Dirichlet BC values in RHS
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            ksp.solve(b, u_sol_vec)
            its = ksp.getIterationNumber()
            total_linear_iterations += its
            
            # Update previous solution
            u_n.x.array[:] = u.x.array
        
        # Compute L2 norm of solution at final time
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value if norm_value > 0 else 0
            if rank == 0:
                print(f"  N={N}, norm={norm_value:.6e}, rel_error={relative_error:.6e}")
            if relative_error < 0.01:
                u_sol_final = u
                mesh_resolution_used = N
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
        else:
            if rank == 0:
                print(f"  N={N}, norm={norm_value:.6e}")
        
        norm_old = norm_value
        u_sol_final = u
        mesh_resolution_used = N
    
    # If loop finished without break, use the last result (N=128)
    if mesh_resolution_used is None:
        mesh_resolution_used = 128
    
    # Prepare output: evaluate solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T  # 3D points
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather on rank 0
    u_all = comm.gather(u_values, root=0)
    if rank == 0:
        u_combined = np.concatenate([arr for arr in u_all if arr is not None])
        # Fill any remaining NaN with 0 (should not happen)
        u_combined = np.nan_to_num(u_combined, nan=0.0)
        u_grid = u_combined.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    # Also get initial condition on grid for optional output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.exp(0) * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
    
    points_on_proc_init = []
    cells_on_proc_init = []
    eval_map_init = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc_init.append(points[i])
            cells_on_proc_init.append(links[0])
            eval_map_init.append(i)
    
    u_init_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc_init) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc_init), np.array(cells_on_proc_init, dtype=np.int32))
        u_init_values[eval_map_init] = vals_init.flatten()
    
    u_init_all = comm.gather(u_init_values, root=0)
    if rank == 0:
        u_init_combined = np.concatenate([arr for arr in u_init_all if arr is not None])
        u_init_combined = np.nan_to_num(u_init_combined, nan=0.0)
        u_initial_grid = u_init_combined.reshape((nx, ny))
    else:
        u_initial_grid = np.zeros((nx, ny), dtype=ScalarType)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    # Return dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
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
    if MPI.COMM_WORLD.rank == 0:
        print("Solver completed successfully")
        print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
        print(f"Solution shape: {result['u'].shape}")

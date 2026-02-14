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
    Solve the heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract time parameters from case_spec or use defaults
    # According to task description: if t_end or dt mentioned, force is_transient=True
    # and set hardcoded defaults
    t_end = 0.12
    dt = 0.02
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Define the source term function
    def source_term(x):
        return np.sin(5*np.pi*x[0]) * np.sin(3*np.pi*x[1]) + \
               0.5 * np.sin(9*np.pi*x[0]) * np.sin(7*np.pi*x[1])
    
    # Define boundary condition (Dirichlet, zero on all boundaries)
    def boundary_marker(x):
        # Mark all boundaries
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []  # Will store (u_sol, domain, V) tuples
    norms = []
    solver_infos = []
    
    for i, N in enumerate(resolutions):
        if rank == 0:
            print(f"Solving with mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        tdim = domain.topology.dim
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Boundary conditions
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        f = fem.Function(V)
        f.interpolate(source_term)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(lambda x: np.zeros_like(x[0]))  # Initial condition u0 = 0.0
        
        # Time-stepping forms (Backward Euler)
        a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (time-independent part)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        for solver_try in range(2):  # Try twice: first iterative, then direct
            try:
                solver = PETSc.KSP().create(domain.comm)
                solver.setOperators(A)
                
                if solver_try == 0:
                    # Iterative solver
                    solver.setType(PETSc.KSP.Type.GMRES)
                    solver.getPC().setType(PETSc.PC.Type.HYPRE)
                    ksp_type = 'gmres'
                    pc_type = 'hypre'
                else:
                    # Direct solver fallback
                    solver.setType(PETSc.KSP.Type.PREONLY)
                    solver.getPC().setType(PETSc.PC.Type.LU)
                    ksp_type = 'preonly'
                    pc_type = 'lu'
                
                solver.setTolerances(rtol=rtol, max_it=1000)
                solver.setFromOptions()
                
                # Time-stepping loop
                total_iterations = 0
                t = 0.0
                n_steps = 0
                
                # Handle time steps carefully to reach exactly t_end
                while t < t_end - 1e-12:
                    # Adjust dt for last step if needed
                    current_dt = min(dt, t_end - t)
                    
                    # Update forms with current dt if it changed
                    if abs(current_dt - dt) > 1e-12:
                        a = ufl.inner(u, v) * ufl.dx + current_dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
                        L = ufl.inner(u_n, v) * ufl.dx + current_dt * ufl.inner(f, v) * ufl.dx
                        a_form = fem.form(a)
                        L_form = fem.form(L)
                        # Reassemble matrix with new dt
                        A = petsc.assemble_matrix(a_form, bcs=[bc])
                        A.assemble()
                        solver.setOperators(A)
                    
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
                    its = solver.getIterationNumber()
                    total_iterations += its
                    
                    # Update for next step
                    u_n.x.array[:] = u_sol.x.array
                    
                    t += current_dt
                    n_steps += 1
                
                solver_success = True
                break  # Success, exit solver try loop
                
            except Exception as e:
                if rank == 0:
                    print(f"Solver try {solver_try} failed: {e}")
                if solver_try == 0:
                    continue  # Try direct solver
                else:
                    raise  # Both failed, re-raise
        
        if not solver_success:
            raise RuntimeError("All solver attempts failed")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        solutions.append((u_sol, domain, V))
        
        # Store solver info for this resolution
        solver_info = {
            "mesh_resolution": N,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,  # Report the nominal dt, not the adjusted one
            "n_steps": n_steps,
            "time_scheme": time_scheme
        }
        solver_infos.append(solver_info)
        
        # Check convergence (compare with previous resolution if available)
        if i > 0:
            relative_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] > 0 else 0.0
            if rank == 0:
                print(f"Relative error between N={resolutions[i-1]} and N={N}: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"Converged at N={N}")
                break
    
    # Use the last solution (either converged or finest mesh)
    final_u_sol, final_domain, final_V = solutions[-1]
    final_solver_info = solver_infos[-1]
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D coordinates even for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    u_grid_flat = evaluate_at_points(final_u_sol, final_domain, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Also get initial condition on same grid
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: np.zeros_like(x[0]))
    u0_grid_flat = evaluate_at_points(u0_func, final_domain, points)
    u0_grid = u0_grid_flat.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": final_solver_info
    }

def evaluate_at_points(u_func, domain, points):
    """
    Evaluate a function at given points.
    points: shape (3, N) numpy array
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we need to gather results from all processes
    comm = domain.comm
    rank = comm.rank
    size = comm.size
    
    if size > 1:
        # Gather all values to root process
        all_values = comm.gather(u_values, root=0)
        if rank == 0:
            # Combine values from all processes (taking first non-NaN value for each point)
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        # Broadcast final values to all processes
        u_values = comm.bcast(u_values, root=0)
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.12,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

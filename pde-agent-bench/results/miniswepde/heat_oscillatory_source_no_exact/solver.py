import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve transient heat equation with adaptive mesh refinement and runtime auto-tuning.
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec with defaults
    # Problem Description explicitly mentions t_end and dt, so force is_transient = True
    t_end = 0.12
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Coefficients
    kappa = 0.8
    if 'coefficients' in case_spec:
        kappa = case_spec['coefficients'].get('kappa', kappa)
    
    # Source term function
    def source_term(x):
        return np.sin(6*np.pi*x[0]) * np.sin(6*np.pi*x[1])
    
    # Initial condition
    def u0_expr(x):
        return np.zeros_like(x[0])
    
    # Boundary condition (Dirichlet, zero on all boundaries)
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
    element_degree = 1  # Start with P1 elements
    
    # Storage for convergence check
    prev_norm = None
    u_final = None
    mesh_resolution_used = None
    solver_info_base = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        tdim = domain.topology.dim
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define functions
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step (unknown)
        
        # Interpolate initial condition
        u_n.interpolate(u0_expr)
        u.interpolate(u0_expr)
        
        # Boundary conditions
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem for backward Euler
        v = ufl.TestFunction(V)
        dt = fem.Constant(domain, ScalarType(dt_suggested))
        kappa_const = fem.Constant(domain, ScalarType(kappa))
        
        # Source term as a function that can be updated
        f_func = fem.Function(V)
        f_func.interpolate(source_term)
        
        # Time-stepping form: (u - u_n)/dt * v dx + kappa * grad(u)·grad(v) dx = f * v dx
        # Rearranged: u*v dx + dt*kappa*grad(u)·grad(v) dx = u_n*v dx + dt*f*v dx
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * kappa_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors - use the function space from L_form
        b = petsc.create_vector(L_form.function_spaces)
        
        # Solver setup with robustness: try iterative first, fallback to direct
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=rtol, max_it=1000)
            solver.setFromOptions()
            
            # Time-stepping loop
            t = 0.0
            n_steps = 0
            total_iterations = 0
            
            while t < t_end - 1e-12:
                # Update time
                t += dt_suggested
                n_steps += 1
                
                # Assemble RHS
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                
                # Apply lifting and BCs
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Solve
                solver.solve(b, u.x.petsc_vec)
                u.x.scatter_forward()
                
                # Get iteration count
                its = solver.getIterationNumber()
                total_iterations += its
                
                # Update for next step
                u_n.x.array[:] = u.x.array
                
            # Success with iterative solver
            solver_info = {
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": total_iterations,
                "dt": dt_suggested,
                "n_steps": n_steps,
                "time_scheme": scheme
            }
            
        except Exception as e:
            # Fallback to direct solver
            print(f"Iterative solver failed for N={N}: {e}. Switching to direct solver.")
            solver.destroy()
            
            # Recreate solver with direct method
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            
            # Reset time-stepping
            u_n.interpolate(u0_expr)
            u.interpolate(u0_expr)
            
            t = 0.0
            n_steps = 0
            total_iterations = 0
            
            while t < t_end - 1e-12:
                t += dt_suggested
                n_steps += 1
                
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                solver.solve(b, u.x.petsc_vec)
                u.x.scatter_forward()
                
                its = solver.getIterationNumber()
                total_iterations += its
                
                u_n.x.array[:] = u.x.array
            
            ksp_type = 'preonly'
            pc_type = 'lu'
            solver_info = {
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": 1e-12,
                "iterations": total_iterations,
                "dt": dt_suggested,
                "n_steps": n_steps,
                "time_scheme": scheme
            }
        
        # Compute L2 norm for convergence check
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                u_final = u
                mesh_resolution_used = N
                solver_info_base = solver_info
                break
        
        prev_norm = norm_value
        u_final = u
        mesh_resolution_used = N
        solver_info_base = solver_info
    
    # If loop finished without break, use the finest mesh result
    if u_final is None:
        # This shouldn't happen, but as safety
        u_final = u
        mesh_resolution_used = resolutions[-1]
    
    # Get domain and function space from the final solution
    V_final = u_final.function_space
    domain_final = V_final.mesh
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D coordinates)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate function at points
    from dolfinx import geometry
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
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also get initial condition on same grid for optional output
    u0_func = fem.Function(V_final)
    u0_func.interpolate(u0_expr)
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    u0_grid = u0_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info_base
    }

# Test if run as script
if __name__ == "__main__":
    # Simple test with minimal case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.12,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "coefficients": {
            "kappa": 0.8
        }
    }
    result = solve(case_spec)
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

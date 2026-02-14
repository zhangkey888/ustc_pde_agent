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
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters
    # Force is_transient = True since problem description mentions t_end and dt
    is_transient = True
    t_end = 0.1
    dt = 0.02
    scheme = 'backward_euler'
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        if 't_end' in time_spec:
            t_end = float(time_spec['t_end'])
        if 'dt' in time_spec:
            dt = float(time_spec['dt'])
        if 'scheme' in time_spec:
            scheme = time_spec['scheme']
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Start with linear elements
    
    # Storage for convergence check
    prev_norm = None
    final_solution = None
    final_mesh_res = None
    final_info = {}
    
    # Track total linear iterations across all meshes and time steps
    total_linear_iterations = 0
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define coefficient kappa (variable)
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 0.6 * ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
        kappa = fem.Function(V)
        kappa.interpolate(lambda x: 1.0 + 0.6 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
        
        # Define source term f
        f_expr = 1.0 + ufl.sin(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])
        f = fem.Function(V)
        f.interpolate(lambda x: 1.0 + np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
        
        # Define initial condition u0
        u0_expr = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        u0 = fem.Function(V)
        u0.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Define boundary condition (Dirichlet, zero on all boundaries)
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
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u_n.x.array[:] = u0.x.array.copy()
        u_n.x.scatter_forward()
        
        u_sol = fem.Function(V)  # Current solution
        
        # Define variational forms for backward Euler
        dt_constant = fem.Constant(domain, ScalarType(dt))
        
        # Mass term
        m = ufl.inner(u, v) * ufl.dx
        # Stiffness term
        a = dt_constant * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        # Combined left-hand side
        a_total = m + a
        # Right-hand side
        L = ufl.inner(u_n, v) * ufl.dx + dt_constant * ufl.inner(f, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a_total)
        L_form = fem.form(L)
        
        # Assemble matrix (time-independent for backward Euler with constant dt)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol_vec = u_sol.x.petsc_vec
        
        # Setup linear solver with iterative first, fallback to direct
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        linear_iterations_this_mesh = 0
        
        for solver_attempt in range(2):  # Try iterative, then direct
            try:
                ksp = PETSc.KSP().create(comm)
                ksp.setOperators(A)
                ksp.setType(ksp_type)
                ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                pc = ksp.getPC()
                pc.setType(pc_type)
                
                # Time-stepping loop
                n_steps = int(np.ceil(t_end / dt))
                actual_steps = 0
                t = 0.0
                
                for step in range(n_steps):
                    if t + dt > t_end:
                        dt_step = t_end - t
                    else:
                        dt_step = dt
                    
                    dt_constant.value = dt_step
                    
                    # Reassemble matrix if dt changed (unlikely here but for robustness)
                    if step == 0 or dt_step != dt:
                        A.zeroEntries()
                        petsc.assemble_matrix(A, a_form, bcs=[bc])
                        A.assemble()
                    
                    # Assemble RHS
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Solve linear system
                    ksp.solve(b, u_sol_vec)
                    u_sol.x.scatter_forward()
                    
                    # Get iteration count
                    its = ksp.getIterationNumber()
                    linear_iterations_this_mesh += its
                    
                    # Update for next step
                    u_n.x.array[:] = u_sol.x.array.copy()
                    u_n.x.scatter_forward()
                    
                    t += dt_step
                    actual_steps += 1
                    
                    if t >= t_end - 1e-12:
                        break
                
                solver_success = True
                total_linear_iterations += linear_iterations_this_mesh
                break  # Success, exit solver attempt loop
                
            except Exception as e:
                if rank == 0:
                    print(f"Solver attempt with {ksp_type}/{pc_type} failed: {e}")
                if solver_attempt == 0:
                    # Fallback to direct solver
                    ksp_type = 'preonly'
                    pc_type = 'lu'
                    rtol = 1e-12
                else:
                    raise
        
        if not solver_success:
            if rank == 0:
                print(f"All solvers failed for N={N}")
            continue
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        if rank == 0:
            print(f"  N={N}: L2 norm = {norm_value:.6e}")
        
        # Check convergence
        if prev_norm is not None:
            rel_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 0.0
            if rank == 0:
                print(f"  Relative error vs previous: {rel_error:.6e}")
            
            if rel_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Convergence achieved at N={N}")
                final_solution = u_sol
                final_mesh_res = N
                final_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": total_linear_iterations,  # Accumulated across all meshes
                    "dt": dt,
                    "n_steps": actual_steps,
                    "time_scheme": scheme
                }
                break
        
        prev_norm = norm_value
        
        # Store as fallback if we reach the end
        final_solution = u_sol
        final_mesh_res = N
        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_linear_iterations,  # Accumulated across all meshes
            "dt": dt,
            "n_steps": actual_steps,
            "time_scheme": scheme
        }
    
    # If loop finished without convergence, use the last (finest) mesh
    if final_solution is None and rank == 0:
        print("Warning: No solution computed")
        return {}
    
    # Sample solution on a 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D even for 2D mesh)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(final_solution, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Evaluate initial condition at points for u_initial
    u0_grid_flat = evaluate_function_at_points(u0, points)
    u0_grid = u0_grid_flat.reshape((nx, ny))
    
    # Prepare result dictionary
    result = {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": final_info
    }
    
    return result

def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    points: shape (3, N) numpy array
    Returns: shape (N,) numpy array of function values
    """
    domain = u_func.function_space.mesh
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
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a minimal case specification
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
    if result:
        print("Solver completed successfully")
        print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Time steps: {result['solver_info']['n_steps']}")
        print(f"Total linear iterations: {result['solver_info']['iterations']}")

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
    # Start timing
    start_time = time.time()
    
    # Extract problem parameters
    # Use hardcoded defaults as per instructions
    t_end = 0.1
    dt = 0.01
    scheme = 'backward_euler'
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        scheme = time_params.get('scheme', scheme)
    
    # Accuracy tolerance
    accuracy_tol = 2.94e-03
    
    # Manufactured solution and source term
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    def source_term(x, t):
        """Compute f = ∂u/∂t - ∇·(κ∇u) analytically."""
        # u = exp(-t)*sin(2πx)*sin(2πy)
        u = np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
        
        # ∂u/∂t = -u
        du_dt = -u
        
        # κ = 1 + 0.3*cos(2πx)*cos(2πy)
        kappa = 1.0 + 0.3 * np.cos(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
        
        # ∇u components
        du_dx = 2*np.pi * np.exp(-t) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
        du_dy = 2*np.pi * np.exp(-t) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
        
        # ∂κ/∂x and ∂κ/∂y
        dkappa_dx = -0.3 * 2*np.pi * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
        dkappa_dy = -0.3 * 2*np.pi * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
        
        # ∂²u/∂x² and ∂²u/∂y²
        d2u_dx2 = -(2*np.pi)**2 * np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
        d2u_dy2 = -(2*np.pi)**2 * np.exp(-t) * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
        
        # ∇·(κ∇u) = κ∇²u + ∇κ·∇u
        # ∇²u = ∂²u/∂x² + ∂²u/∂y²
        laplacian_u = d2u_dx2 + d2u_dy2
        
        # ∇κ·∇u = ∂κ/∂x * ∂u/∂x + ∂κ/∂y * ∂u/∂y
        grad_kappa_dot_grad_u = dkappa_dx * du_dx + dkappa_dy * du_dy
        
        div_kappa_grad_u = kappa * laplacian_u + grad_kappa_dot_grad_u
        
        # f = ∂u/∂t - ∇·(κ∇u)
        f = du_dt - div_kappa_grad_u
        return f
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    
    # Store results for each resolution
    results = []
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        element_degree = 1  # Using linear elements for speed
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u_np1 = fem.Function(V)  # Current time step
        
        # Initial condition
        def u0_expr(x):
            return exact_solution(x, 0.0)
        u_n.interpolate(u0_expr)
        
        # Variable kappa coefficient as UFL expression
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 0.3 * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])
        
        # Create kappa function for interpolation
        kappa_func = fem.Function(V)
        kappa_func.interpolate(lambda x: 1.0 + 0.3 * np.cos(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]))
        
        # Source term function
        f_func = fem.Function(V)
        
        # Boundary conditions (Dirichlet from exact solution)
        def boundary_marker(x):
            # Apply on entire boundary
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
        def update_bc(t):
            u_bc.interpolate(lambda x: exact_solution(x, t))
        
        update_bc(0.0)  # Initial BC at t=0
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Time-stepping
        t = 0.0
        n_steps = int(t_end / dt)
        
        # Variational form for backward Euler
        a = ufl.inner(u, v) * ufl.dx + dt * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (time-independent)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct
        ksp = PETSc.KSP().create(domain.comm)
        ksp_type = 'gmres'
        pc_type = 'hypre'
        
        try:
            # Configure iterative solver
            ksp.setOperators(A)
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.getPC().setType(PETSc.PC.Type.HYPRE)
            ksp.setTolerances(rtol=1e-8, max_it=1000)
            ksp.setFromOptions()
            
            # Test the solver
            test_vec = b.duplicate()
            test_vec.setRandom()
            ksp.solve(test_vec, test_vec)
            
        except Exception as e:
            # Fallback to direct solver
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.LU)
            ksp_type = 'preonly'
            pc_type = 'lu'
        
        # Time stepping
        total_iterations = 0
        for step in range(n_steps):
            t += dt
            
            # Update boundary condition for current time
            update_bc(t)
            
            # Update source term for current time
            f_func.interpolate(lambda x: source_term(x, t))
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            ksp.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            
            # Get iteration count
            total_iterations += ksp.getIterationNumber()
            
            # Update previous solution
            u_n.x.array[:] = u_sol.x.array
        
        # Compute error at final time
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: exact_solution(x, t_end))
        
        # Compute L2 error
        error_expr = ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx
        error_form = fem.form(error_expr)
        error = np.sqrt(fem.assemble_scalar(error_form))
        
        # Compute norm of solution for convergence check
        norm_expr = ufl.inner(u_sol, u_sol) * ufl.dx
        norm_form = fem.form(norm_expr)
        norm = np.sqrt(fem.assemble_scalar(norm_form))
        
        results.append({
            'N': N,
            'u_sol': u_sol,
            'domain': domain,
            'error': error,
            'norm': norm,
            'iterations': total_iterations,
            'ksp_type': ksp_type,
            'pc_type': pc_type
        })
        
        print(f"Mesh N={N}, L2 error={error:.2e}, norm={norm:.2e}")
        
        # Check if error meets accuracy requirement
        if error <= accuracy_tol:
            print(f"Accuracy requirement met at N={N} with error={error:.2e}")
            break
    
    # Use the last (best) result
    result = results[-1]
    final_u = result['u_sol']
    final_domain = result['domain']
    final_error = result['error']
    final_N = result['N']
    
    # Solver info
    solver_info = {
        'mesh_resolution': final_N,
        'element_degree': element_degree,
        'ksp_type': result['ksp_type'],
        'pc_type': result['pc_type'],
        'rtol': 1e-8,
        'iterations': result['iterations'],
        'dt': dt,
        'n_steps': n_steps,
        'time_scheme': scheme,
        'nonlinear_iterations': []
    }
    
    # Create 50x50 uniform grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Points for evaluation (shape (3, nx*ny))
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
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also compute initial condition on same grid
    V_final = fem.functionspace(final_domain, ("Lagrange", element_degree))
    u0_func = fem.Function(V_final)
    u0_func.interpolate(lambda x: exact_solution(x, 0.0))
    
    # Evaluate initial condition
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    u_initial = u0_values.reshape((nx, ny))
    
    # End timing
    end_time = time.time()
    solver_info['wall_time'] = end_time - start_time
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a minimal case specification
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
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Max solution value: {np.max(result['u'])}")
    print(f"Min solution value: {np.min(result['u'])}")

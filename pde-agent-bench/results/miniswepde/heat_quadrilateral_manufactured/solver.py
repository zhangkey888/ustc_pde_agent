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
    Returns dict with 'u' (solution array) and 'solver_info'.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters with defaults
    pde_info = case_spec.get('pde', {})
    time_info = pde_info.get('time', {})
    
    # Time parameters - use provided or defaults from problem description
    t_end = time_info.get('t_end', 0.1)
    dt = time_info.get('dt', 0.01)
    scheme = time_info.get('scheme', 'backward_euler')
    
    # Coefficient
    kappa = pde_info.get('coefficients', {}).get('kappa', 1.0)
    
    # Domain - unit square
    domain_bounds = case_spec.get('domain', {}).get('bounds', [[0,1], [0,1]])
    p0 = np.array([domain_bounds[0][0], domain_bounds[1][0]])
    p1 = np.array([domain_bounds[0][1], domain_bounds[1][1]])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    
    # For tracking convergence
    prev_norm = None
    final_solution = None
    final_mesh_res = None
    final_domain = None
    final_V = None
    
    total_linear_iterations = 0
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps  # Adjust dt to exactly reach t_end
    
    pi = np.pi
    
    for N in resolutions:
        print(f"Testing mesh resolution N={N}")
        # Create mesh
        domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                       cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet from exact solution)
        def boundary_marker(x):
            # All boundaries
            return np.ones(x.shape[1], dtype=bool)
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create a BC for matrix assembly (value doesn't matter for matrix)
        u_bc_dummy = fem.Function(V)
        u_bc_dummy.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc_dummy, dofs)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step
        
        # Initial condition at t=0: u0 = sin(pi*x)*sin(pi*y)
        u_n.interpolate(lambda x: np.sin(pi*x[0]) * np.sin(pi*x[1]))
        u.x.array[:] = u_n.x.array
        
        # Define variational problem for backward Euler
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        
        # Weak form: (u - u_n)/dt * v dx + kappa * grad(u)·grad(v) dx = f * v dx
        a = (1/actual_dt) * ufl.inner(u_trial, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L_base = (1/actual_dt) * ufl.inner(u_n, v) * ufl.dx
        
        # Create forms
        a_form = fem.form(a)
        L_base_form = fem.form(L_base)
        
        # Form for source term f
        f_func = fem.Function(V)  # Will update each time step
        
        # Assemble matrix with BCs applied
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_base_form.function_spaces)
        
        # Try iterative solver first, fallback to direct
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        for solver_try in range(2):  # Try twice: iterative then direct
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            
            if solver_try == 0:
                # Iterative solver
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                ksp_type = 'gmres'
                pc_type = 'hypre'
            else:
                # Direct solver fallback
                solver.setType(PETSc.KSP.Type.PREONLY)
                solver.getPC().setType(PETSc.PC.Type.LU)
                ksp_type = 'preonly'
                pc_type = 'lu'
            
            # Time stepping loop
            current_linear_iterations = 0
            try:
                for step in range(n_steps):
                    t = (step + 1) * actual_dt
                    
                    # Update boundary condition value at current time
                    u_bc = fem.Function(V)
                    u_bc.interpolate(lambda x: np.exp(-t) * np.sin(pi*x[0]) * np.sin(pi*x[1]))
                    # Create new BC with same dofs but updated value
                    bc_current = fem.dirichletbc(u_bc, dofs)
                    
                    # Update source term f at current time
                    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(2*pi**2 - 1) for kappa=1
                    f_func.interpolate(lambda x: np.exp(-t) * (2*pi**2 - 1) * np.sin(pi*x[0]) * np.sin(pi*x[1]))
                    
                    # Reassemble RHS: b = (1/dt)*u_n*v dx + f*v dx
                    with b.localForm() as loc:
                        loc.set(0)
                    
                    # Assemble (1/dt)*u_n*v dx part
                    petsc.assemble_vector(b, L_base_form)
                    
                    # Add f*v dx term
                    f_form = fem.form(ufl.inner(f_func, v) * ufl.dx)
                    b_f = petsc.assemble_vector(f_form)
                    b.axpy(1.0, b_f)
                    b_f.destroy()
                    
                    # Apply boundary conditions to RHS
                    petsc.apply_lifting(b, [a_form], bcs=[[bc_current]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc_current])
                    
                    # Solve linear system
                    solver.solve(b, u.x.petsc_vec)
                    u.x.scatter_forward()
                    
                    # Get iteration count
                    current_linear_iterations += solver.getIterationNumber()
                    
                    # Update for next step
                    u_n.x.array[:] = u.x.array
                
                solver_success = True
                total_linear_iterations += current_linear_iterations
                break  # Exit solver try loop
                
            except Exception as e:
                if solver_try == 0:
                    print(f"Iterative solver failed for N={N}, trying direct solver: {e}")
                    continue
                else:
                    print(f"Direct solver also failed for N={N}: {e}")
                    raise
        
        if not solver_success:
            raise RuntimeError(f"All solvers failed for N={N}")
        
        # Compute L2 norm of solution at final time
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        print(f"  N={N}, L2 norm = {norm_value:.6e}")
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            print(f"  Relative error vs previous: {relative_error:.6e}")
            if relative_error < 0.01:  # 1% convergence
                print(f"  CONVERGED at N={N}")
                final_solution = u
                final_mesh_res = N
                final_domain = domain
                final_V = V
                break
        
        prev_norm = norm_value
        final_solution = u
        final_mesh_res = N
        final_domain = domain
        final_V = V
    
    # If loop completes without break, use finest mesh (N=128)
    if final_solution is None:
        final_solution = u
        final_mesh_res = 128
        final_domain = domain
        final_V = V
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(p0[0], p1[0], nx)
    y_vals = np.linspace(p0[1], p1[1], ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points for evaluation (3D coordinates)
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
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    # Also get initial condition for optional output
    u_initial_func = fem.Function(final_V)
    u_initial_func.interpolate(lambda x: np.sin(pi*x[0]) * np.sin(pi*x[1]))
    u_initial_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_values.reshape(nx, ny)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_linear_iterations,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        'pde': {
            'type': 'heat',
            'time': {
                't_end': 0.1,
                'dt': 0.01,
                'scheme': 'backward_euler'
            },
            'coefficients': {
                'kappa': 1.0
            }
        },
        'domain': {
            'type': 'unit_square',
            'bounds': [[0,1], [0,1]]
        }
    }
    
    result = solve(case_spec)
    print("\nTest completed successfully")
    print(f"u shape: {result['u'].shape}")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Total linear iterations: {result['solver_info']['iterations']}")

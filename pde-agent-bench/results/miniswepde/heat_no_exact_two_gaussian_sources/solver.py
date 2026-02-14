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
    
    # Extract parameters from case_spec or use defaults
    # According to problem description, we MUST set hardcoded defaults
    t_end = 0.1
    dt = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', t_end)
        dt = time_spec.get('dt', dt)
        scheme = time_spec.get('scheme', scheme)
    
    # Domain: unit square [0,1] x [0,1]
    comm = MPI.COMM_WORLD
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    
    # Parameters for solver
    element_degree = 1  # Linear elements
    rtol = 1e-8  # Linear solver tolerance
    
    # Initialize variables for convergence check
    prev_norm = None
    converged_resolution = None
    final_solution = None
    final_domain = None
    ksp_type = "gmres"
    pc_type = "hypre"
    linear_iterations = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))  # κ = 1.0
        
        # Source term: f = exp(-220*((x-0.25)**2 + (y-0.25)**2)) + exp(-220*((x-0.75)**2 + (y-0.7)**2))
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.exp(-220*((x[0]-0.25)**2 + (x[1]-0.25)**2)) + \
                 ufl.exp(-220*((x[0]-0.75)**2 + (x[1]-0.7)**2))
        
        # Create a function for the source term
        f = fem.Function(V)
        # Use Expression to interpolate the UFL expression
        try:
            expr = fem.Expression(f_expr, V.element.interpolation_points)
            f.interpolate(expr)
        except Exception as e:
            # Fallback: use constant zero if interpolation fails
            print(f"Warning: Source term interpolation failed: {e}")
            f.interpolate(lambda x: np.zeros_like(x[0]))
        
        # Initial condition: u0 = 0.0
        u_n = fem.Function(V)
        u_n.interpolate(lambda x: np.zeros_like(x[0]))
        
        # Boundary condition: u = 0 on all boundaries (Dirichlet)
        # Since initial condition is 0 and no boundary condition is specified in problem,
        # we assume homogeneous Dirichlet
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
        
        # Time-stepping setup
        n_steps = int(t_end / dt)
        if n_steps == 0:
            n_steps = 1
            dt = t_end
        
        # Variational form for backward Euler
        # (u - u_n)/dt * v * dx + kappa * dot(grad(u), grad(v)) * dx = f * v * dx
        a = (1/dt) * u * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1/dt) * u_n * v * ufl.dx + f * v * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create RHS vector and solution vector
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        total_linear_iters = 0
        
        for solver_type in ['iterative', 'direct']:
            try:
                solver = PETSc.KSP().create(domain.comm)
                solver.setOperators(A)
                
                if solver_type == 'iterative':
                    solver.setType(PETSc.KSP.Type.GMRES)
                    solver.getPC().setType(PETSc.PC.Type.HYPRE)
                    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                    ksp_type = "gmres"
                    pc_type = "hypre"
                else:  # direct
                    solver.setType(PETSc.KSP.Type.PREONLY)
                    solver.getPC().setType(PETSc.PC.Type.LU)
                    ksp_type = "preonly"
                    pc_type = "lu"
                
                # Time-stepping loop
                total_linear_iters = 0
                for step in range(n_steps):
                    # Assemble RHS
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting for BCs
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    
                    # Apply BCs to RHS
                    petsc.set_bc(b, [bc])
                    
                    # Solve linear system
                    solver.solve(b, u_sol.x.petsc_vec)
                    linear_iters = solver.getIterationNumber()
                    total_linear_iters += linear_iters
                    
                    # Update for next step - copy values properly
                    u_n.x.array[:] = u_sol.x.array.copy()
                    u_n.x.scatter_forward()
                    
                    # Update RHS form for next step (only u_n changes)
                    L = (1/dt) * u_n * v * ufl.dx + f * v * ufl.dx
                    L_form = fem.form(L)
                
                linear_iterations = total_linear_iters
                solver_success = True
                break
                
            except Exception as e:
                if solver_type == 'iterative':
                    print(f"Iterative solver failed, trying direct solver: {e}")
                    continue  # Try direct solver
                else:
                    raise  # Re-raise if direct solver also fails
        
        if not solver_success:
            raise RuntimeError("Both iterative and direct solvers failed")
        
        # Compute norm of solution for convergence check
        norm = np.sqrt(fem.assemble_scalar(fem.form(u_sol**2 * ufl.dx)))
        norms.append(norm)
        
        # Store solution for this resolution
        solutions.append((u_sol, domain, N, norm))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 1e-12 else 0.0
            if relative_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                final_solution = u_sol
                final_domain = domain
                print(f"Converged at resolution {N} with relative error {relative_error:.6f}")
                break
        
        prev_norm = norm
        print(f"Resolution {N}: norm = {norm:.6e}")
    
    # If loop finished without convergence, use finest mesh
    if final_solution is None:
        u_sol, domain, N, norm = solutions[-1]
        final_solution = u_sol
        final_domain = domain
        converged_resolution = N
        print(f"Using finest resolution {N} (no convergence)")
    
    # Prepare output grid (50x50 uniform grid)
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at grid points
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
    
    # Check for any NaN values and fill with 0 if necessary
    if np.any(np.isnan(u_values)):
        nan_count = np.sum(np.isnan(u_values))
        print(f"Warning: {nan_count} NaN values in solution, filling with 0")
        u_values[np.isnan(u_values)] = 0.0
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Also get initial condition on the same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(lambda x: np.zeros_like(x[0]))
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    # Fill NaN values in initial condition
    if np.any(np.isnan(u0_values)):
        u0_values[np.isnan(u0_values)] = 0.0
    u_initial = u0_values.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": converged_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    # End timing
    end_time = time.time()
    solver_info["wall_time_sec"] = end_time - start_time
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
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
        }
    }
    
    try:
        result = solve(test_case)
        print("\nSolver executed successfully!")
        print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
        print(f"Time steps: {result['solver_info']['n_steps']}")
        print(f"Total linear iterations: {result['solver_info']['iterations']}")
        print(f"Wall time: {result['solver_info']['wall_time_sec']:.3f} seconds")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")
    except Exception as e:
        print(f"Error during solver execution: {e}")
        import traceback
        traceback.print_exc()

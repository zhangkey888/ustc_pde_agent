import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
from dolfinx import nls
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    # Problem description explicitly mentions t_end=0.1 and dt=0.02
    # Force is_transient = True as fallback
    t_end = 0.1
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if case_spec is not None and 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Define source term and initial condition functions
    def source_term_func(x):
        return np.exp(-200 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    def initial_condition_func(x):
        return np.exp(-120 * ((x[0] - 0.6)**2 + (x[1] - 0.4)**2))
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Storage for convergence checking
    prev_norm = None
    final_solution = None
    final_mesh_resolution = None
    final_element_degree = 1  # Using linear elements for speed
    final_domain = None
    final_V = None
    
    # Solver info to be populated
    solver_info = {
        "mesh_resolution": None,
        "element_degree": final_element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt_suggested,
        "n_steps": int(np.ceil(t_end / dt_suggested)),
        "time_scheme": scheme,
        "nonlinear_iterations": []  # Not needed for linear problem
    }
    
    total_linear_iterations = 0
    used_direct_solver = False
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", final_element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define functions for current and previous time steps
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(initial_condition_func)
        
        # Source term as a Function (interpolated)
        f = fem.Function(V)
        f.interpolate(source_term_func)
        
        # Current solution
        u_current = fem.Function(V)
        
        # Boundary condition (Dirichlet, zero on all boundaries)
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
        
        # Zero Dirichlet BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Time-stepping parameters
        dt = dt_suggested
        n_steps = int(np.ceil(t_end / dt))
        if n_steps == 0:
            n_steps = 1
        
        # Define forms for backward Euler
        kappa = fem.Constant(domain, ScalarType(1.0))  # Diffusion coefficient
        
        # Mass matrix term
        a_mass = ufl.inner(u, v) * ufl.dx
        # Stiffness matrix term
        a_stiff = dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        # Combined bilinear form
        a = a_mass + a_stiff
        
        # Linear form
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        
        # Setup linear solver with iterative method first (unless direct was already used)
        if not used_direct_solver:
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=1e-8, max_it=1000)
            solver.setFromOptions()
        else:
            # Use direct solver if previous mesh needed it
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            solver.setFromOptions()
        
        # Time-stepping loop with retry mechanism for stability
        linear_iterations_this_mesh = 0
        current_dt = dt
        
        for step in range(n_steps):
            # Update time
            t = (step + 1) * current_dt
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply lifting and BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve linear system with retry
            max_retries = 2
            for retry in range(max_retries + 1):
                try:
                    solver.solve(b, u_current.x.petsc_vec)
                    u_current.x.scatter_forward()
                    its = solver.getIterationNumber()
                    linear_iterations_this_mesh += its
                    break  # Success
                except Exception as e:
                    if retry < max_retries:
                        # Try with direct solver
                        print(f"Solver failed, retrying with direct solver (retry {retry+1}): {e}")
                        solver_direct = PETSc.KSP().create(domain.comm)
                        solver_direct.setOperators(A)
                        solver_direct.setType(PETSc.KSP.Type.PREONLY)
                        solver_direct.getPC().setType(PETSc.PC.Type.LU)
                        solver_direct.setFromOptions()
                        solver = solver_direct
                        used_direct_solver = True
                        solver_info["ksp_type"] = "preonly"
                        solver_info["pc_type"] = "lu"
                    else:
                        raise  # Re-raise if all retries fail
            
            # Update previous solution
            u_n.x.array[:] = u_current.x.array
        
        total_linear_iterations += linear_iterations_this_mesh
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_current, u_current) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                final_solution = u_current
                final_mesh_resolution = N
                final_domain = domain
                final_V = V
                break
        
        prev_norm = norm_value
        final_solution = u_current
        final_mesh_resolution = N
        final_domain = domain
        final_V = V
    
    # If loop finished without break, use the finest mesh result
    if final_mesh_resolution is None:
        final_mesh_resolution = resolutions[-1]
    
    # Update solver info
    solver_info["mesh_resolution"] = final_mesh_resolution
    solver_info["iterations"] = total_linear_iterations
    solver_info["n_steps"] = int(np.ceil(t_end / dt_suggested))
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    if final_domain is not None:
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
        
        u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        # Reshape to (nx, ny)
        u_grid = u_values.reshape((nx, ny))
        
        # Also compute initial condition on the same grid for u_initial
        # Create a function for initial condition
        u0_func = fem.Function(final_V)
        u0_func.interpolate(initial_condition_func)
        
        # Evaluate initial condition
        u0_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u0_values[eval_map] = vals0.flatten()
        u_initial = u0_values.reshape((nx, ny))
    else:
        # Fallback: return zeros if something went wrong
        u_grid = np.zeros((nx, ny), dtype=ScalarType)
        u_initial = np.zeros((nx, ny), dtype=ScalarType)
    
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
    # Create a minimal case_spec for testing
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
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

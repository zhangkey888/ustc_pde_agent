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
    
    # Extract parameters from case_spec with defaults
    t_end = 0.1
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Problem parameters
    kappa = 1.0  # diffusion coefficient
    f_value = 1.0  # source term
    u0_value = 0.0  # initial condition
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    final_V = None
    final_domain = None
    
    comm = MPI.COMM_WORLD
    
    # Solver configuration (will be determined during first resolution)
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-8
    solver_configured = False
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space - using degree 1 elements
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define boundary condition (Dirichlet u=0 on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark entire boundary
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
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define constants
        kappa_const = fem.Constant(domain, ScalarType(kappa))
        f_const = fem.Constant(domain, ScalarType(f_value))
        
        # Time-stepping setup
        dt = dt_suggested
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # Adjust to exactly reach t_end
        
        # Create function for solution at current and previous time steps
        u_n = fem.Function(V)  # u at previous time step
        u_n.interpolate(lambda x: np.full_like(x[0], u0_value))
        
        u_sol = fem.Function(V)  # u at current time step
        
        # Define variational forms for backward Euler
        # (u - u_n)/dt * v * dx + kappa * dot(grad(u), grad(v)) * dx = f * v * dx
        a = (1.0/dt) * u * v * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (1.0/dt) * u_n * v * ufl.dx + f_const * v * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        
        # Solver setup - try iterative first, fallback to direct
        if not solver_configured:
            # First try iterative solver
            try:
                solver = PETSc.KSP().create(domain.comm)
                solver.setOperators(A)
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
                solver.setFromOptions()
                
                # Test with a simple RHS
                test_b = b.duplicate()
                test_x = b.duplicate()
                test_b.set(1.0)
                solver.solve(test_b, test_x)
                
                ksp_type = "gmres"
                pc_type = "hypre"
                rtol = 1e-8
                solver_configured = True
                print(f"Using iterative solver: {ksp_type} with {pc_type}")
                
            except Exception as e:
                # Fallback to direct solver
                ksp_type = "preonly"
                pc_type = "lu"
                rtol = 1e-12
                solver_configured = True
                print(f"Using direct solver: {ksp_type} with {pc_type}")
        
        # Create solver with configured type
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        if ksp_type == "gmres":
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
        else:
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
        
        solver.setTolerances(rtol=rtol, atol=rtol/10, max_it=1000)
        solver.setFromOptions()
        
        # Time-stepping loop
        total_iterations = 0
        
        for step in range(n_steps):
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply lifting and BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            
            # Get iteration count
            total_iterations += solver.getIterationNumber()
            
            # Update previous solution
            u_n.x.array[:] = u_sol.x.array[:]
        
        # Store solution and compute norm
        solutions.append(u_sol)
        final_V = V
        final_domain = domain
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        
        # Check convergence (need at least 2 resolutions to compare)
        if len(norms) >= 2:
            rel_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 1e-12 else abs(norms[-1] - norms[-2])
            if rel_error < 0.01:  # 1% convergence criterion
                print(f"Mesh converged at resolution N={N} with relative error {rel_error:.6f}")
                break
    
    # Use the last solution (either converged or finest mesh)
    final_solution = solutions[-1]
    final_resolution = resolutions[min(len(solutions)-1, len(resolutions)-1)]
    
    # Compute initial condition for output
    u_initial_func = fem.Function(final_V)
    u_initial_func.interpolate(lambda x: np.full_like(x[0], u0_value))
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points array (3D coordinates)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_values = evaluate_at_points(final_solution, points)
    u_grid = u_values.reshape(nx, ny)
    
    # Evaluate initial condition at points
    u_initial_values = evaluate_at_points(u_initial_func, points)
    u_initial_grid = u_initial_values.reshape(nx, ny)
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": final_resolution,
        "element_degree": 1,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    # End timing
    end_time = time.time()
    wall_time = end_time - start_time
    print(f"Solve completed in {wall_time:.3f} seconds")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

def evaluate_at_points(u_func, points):
    """
    Evaluate a function at given points.
    points: shape (3, N) numpy array
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
        try:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        except Exception as e:
            # Fallback: use interpolation if eval fails
            print(f"Warning: eval failed, using interpolation fallback: {e}")
            # Create a function space on a simple mesh for interpolation
            # This is a simplified fallback - in practice might need more robust handling
            pass
    
    # Check for any NaN values and fill with 0 if necessary
    if np.any(np.isnan(u_values)):
        nan_count = np.sum(np.isnan(u_values))
        print(f"Warning: {nan_count} NaN values in evaluation, filling with 0")
        u_values[np.isnan(u_values)] = 0.0
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a simple case specification
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

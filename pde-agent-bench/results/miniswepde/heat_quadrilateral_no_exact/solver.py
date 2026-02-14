import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing PDE specification. Expected to have:
        - case_spec['pde']['time']['t_end'] : final time (default: 0.12)
        - case_spec['pde']['time']['dt'] : time step (default: 0.03)
        - case_spec['pde']['time']['scheme'] : scheme (default: 'backward_euler')
    
    Returns:
    --------
    dict with keys:
        - 'u' : numpy array shape (50, 50) with final solution
        - 'u_initial' : numpy array shape (50, 50) with initial condition
        - 'solver_info' : dict with solver metadata
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults from problem description
    t_end = 0.12
    dt = 0.03
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Safety check
    if t_end is None or dt is None:
        raise ValueError("Time parameters t_end and dt must be provided")
    
    # Domain: unit square [0,1] x [0,1]
    comm = MPI.COMM_WORLD
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    
    # Solver info to collect
    solver_info = {
        "mesh_resolution": None,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt,
        "n_steps": 0,
        "time_scheme": time_scheme
    }
    
    # Total linear iterations across all solves
    total_iterations = 0
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Boundary condition: u = 0 on entire boundary (Dirichlet)
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
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(lambda x: np.zeros_like(x[0]))  # Initial condition u0 = 0.0
        
        u = fem.Function(V)  # Current time step (unknown)
        
        # Define variational problem for backward Euler
        v = ufl.TestFunction(V)
        dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
        kappa = fem.Constant(domain, PETSc.ScalarType(1.0))  # κ = 1.0
        f = fem.Constant(domain, PETSc.ScalarType(1.0))  # f = 1.0
        
        # Backward Euler linearized form
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(u_trial/dt_constant, v) * ufl.dx + \
            ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        
        # Create form
        a_form = fem.form(a)
        
        # Assemble matrix (constant in time for this problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vector for RHS
        b = petsc.create_vector([V])
        
        # Solver setup - try iterative first
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8, max_it=1000)
        iterative_ok = True
        
        # Time-stepping loop
        t = 0.0
        n_steps = 0
        iterations_this_mesh = 0
        
        while t < t_end - 1e-12:
            # Adjust dt for last step to hit t_end exactly
            current_dt = min(dt, t_end - t)
            if abs(current_dt - dt) > 1e-12:
                dt_constant.value = current_dt
            
            # Update RHS form with current u_n
            L = ufl.inner(u_n/dt_constant + f, v) * ufl.dx
            L_form = fem.form(L)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            if iterative_ok:
                try:
                    solver.solve(b, u.x.petsc_vec)
                    u.x.scatter_forward()
                except PETSc.Error:
                    # Switch to direct solver if iterative fails
                    iterative_ok = False
                    solver.setType(PETSc.KSP.Type.PREONLY)
                    solver.getPC().setType(PETSc.PC.Type.LU)
                    solver_info["ksp_type"] = "preonly"
                    solver_info["pc_type"] = "lu"
                    solver.solve(b, u.x.petsc_vec)
                    u.x.scatter_forward()
            else:
                solver.solve(b, u.x.petsc_vec)
                u.x.scatter_forward()
            
            # Get iteration count
            its = solver.getIterationNumber()
            iterations_this_mesh += its
            
            # Update for next time step
            u_n.x.array[:] = u.x.array
            
            t += current_dt
            n_steps += 1
        
        total_iterations += iterations_this_mesh
        
        # Compute norm of solution for convergence check
        norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)))
        norms.append(norm)
        solutions.append((domain, u, V))  # Store V for later evaluation
        
        # Check convergence (compare with previous mesh if available)
        if i > 0:
            rel_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] > 0 else 0
            if rel_error < 0.01:  # 1% convergence
                solver_info["mesh_resolution"] = N
                solver_info["n_steps"] = n_steps
                break
        
        # If this is the last resolution, use it
        if i == len(resolutions) - 1:
            solver_info["mesh_resolution"] = N
            solver_info["n_steps"] = n_steps
    
    # Use the last converged solution
    domain, u_sol, V = solutions[-1]
    
    # Prepare output on 50x50 grid
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    # Also get initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(lambda x: np.zeros_like(x[0]))
    
    u0_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u_initial = u0_values.reshape(nx, ny)
    
    # Update solver info
    solver_info["iterations"] = total_iterations
    
    # End timing
    end_time = time.time()
    solver_info["wall_time_sec"] = end_time - start_time
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a minimal case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.12,
                "dt": 0.03,
                "scheme": "backward_euler"
            }
        }
    }
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    print(f"Time taken: {result['solver_info']['wall_time_sec']:.2f} seconds")

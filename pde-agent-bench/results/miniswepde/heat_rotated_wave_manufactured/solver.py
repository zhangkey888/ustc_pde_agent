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
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing PDE specification. Expected keys:
        - 'pde': dict with 'time' subdict containing 't_end', 'dt', 'scheme'
        
    Returns
    -------
    dict
        Contains:
        - 'u': numpy array shape (50, 50) with final solution sampled on uniform grid
        - 'u_initial': numpy array shape (50, 50) with initial condition
        - 'solver_info': dict with mesh resolution, element degree, solver parameters,
          time stepping info, and iteration counts.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract parameters from case_spec with defaults
    t_end = 0.1
    dt_suggested = 0.01
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Manufactured solution
    def u_exact(x, t):
        return np.exp(-t) * np.sin(3*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1]))
    
    # Source term: f = (-1 + 20π²) * exp(-t) * sin(π(x-y)) * sin(3π(x+y))
    def source_term(x, t):
        coeff = -1.0 + 20.0 * np.pi**2
        return coeff * np.exp(-t) * np.sin(np.pi*(x[0] - x[1])) * np.sin(3*np.pi*(x[0] + x[1]))
    
    # Adaptive parameters
    resolutions = [32, 64, 128, 256]
    degrees = [1]
    dts = [0.01]
    
    # Target error for domain L2 norm (stricter than grid error requirement)
    target_error = 2.5e-03
    best_solution = None
    best_info = None
    best_error = float('inf')
    
    for degree in degrees:
        for N in resolutions:
            for dt in dts:
                if rank == 0:
                    print(f"Testing N={N}, degree={degree}, dt={dt}")
                
                # Create mesh
                domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
                
                # Function space
                V = fem.functionspace(domain, ("Lagrange", degree))
                
                # Boundary condition: Dirichlet using exact solution
                tdim = domain.topology.dim
                fdim = tdim - 1
                
                def boundary_marker(x):
                    return np.logical_or.reduce([
                        np.isclose(x[0], 0.0),
                        np.isclose(x[0], 1.0),
                        np.isclose(x[1], 0.0),
                        np.isclose(x[1], 1.0)
                    ])
                
                boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
                dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
                
                # Time-dependent boundary condition function
                u_bc = fem.Function(V)
                bc = fem.dirichletbc(u_bc, dofs)
                
                # Initial condition
                u_n = fem.Function(V)
                u_n.interpolate(lambda x: u_exact(x, 0.0))
                
                # Time-stepping
                num_steps = int(np.ceil(t_end / dt))
                if num_steps * dt < t_end - 1e-12:
                    num_steps += 1
                
                # Define variational problem
                u = ufl.TrialFunction(V)
                v = ufl.TestFunction(V)
                
                # Source term as a function
                f = fem.Function(V)
                
                # Time constant
                t_const = fem.Constant(domain, ScalarType(0.0))
                
                # Weak form for backward Euler
                a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
                
                # Assemble forms
                a_form = fem.form(a)
                L_form = fem.form(L)
                
                # Assemble matrix (constant in time)
                A = petsc.assemble_matrix(a_form, bcs=[bc])
                A.assemble()
                
                # Create solver with iterative first, fallback to direct
                solver = PETSc.KSP().create(comm)
                solver.setOperators(A)
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=1e-8, max_it=1000)
                
                # Time-stepping loop
                u = fem.Function(V)
                u.x.array[:] = u_n.x.array
                
                total_iterations = 0
                
                for step in range(num_steps):
                    t = min((step + 1) * dt, t_end)
                    t_const.value = t
                    
                    # Update boundary condition
                    u_bc.interpolate(lambda x: u_exact(x, t))
                    
                    # Update source term
                    f.interpolate(lambda x: source_term(x, t))
                    
                    # Assemble RHS
                    b = petsc.create_vector(L_form.function_spaces)
                    with b.localForm() as loc:
                        loc.set(0)
                    petsc.assemble_vector(b, L_form)
                    
                    # Apply lifting and BCs
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Solve
                    u_sol = fem.Function(V)
                    try:
                        solver.solve(b, u_sol.x.petsc_vec)
                        u_sol.x.scatter_forward()
                        total_iterations += solver.getIterationNumber()
                    except PETSc.Error as e:
                        # Fallback to direct solver
                        if rank == 0:
                            print(f"GMRES failed, switching to direct solver")
                        solver.setType(PETSc.KSP.Type.PREONLY)
                        solver.getPC().setType(PETSc.PC.Type.LU)
                        solver.solve(b, u_sol.x.petsc_vec)
                        u_sol.x.scatter_forward()
                        total_iterations += solver.getIterationNumber()
                    
                    # Update for next step
                    u_n.x.array[:] = u_sol.x.array
                
                # Compute error against exact solution
                u_exact_func = fem.Function(V)
                u_exact_func.interpolate(lambda x: u_exact(x, t_end))
                
                error_expr = ufl.inner(u_n - u_exact_func, u_n - u_exact_func) * ufl.dx
                error_form = fem.form(error_expr)
                error_sq = fem.assemble_scalar(error_form)
                error_sq_global = comm.allreduce(error_sq, op=MPI.SUM)
                l2_error = np.sqrt(error_sq_global)
                
                if rank == 0:
                    print(f"  Domain L2 error = {l2_error:.6e}")
                
                # Store if this is the best solution so far
                if l2_error < best_error:
                    best_error = l2_error
                    best_solution = u_n
                    best_info = {
                        "mesh_resolution": N,
                        "element_degree": degree,
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "rtol": 1e-8,
                        "iterations": total_iterations,
                        "dt": dt,
                        "n_steps": num_steps,
                        "time_scheme": scheme
                    }
                    best_domain = domain
                    best_V = V
                
                # Early exit if error is below target
                if l2_error <= target_error:
                    if rank == 0:
                        print(f"Target accuracy achieved with N={N}, degree={degree}, dt={dt}")
                    break
            if best_error <= target_error:
                break
        if best_error <= target_error:
            break
    
    # If target not achieved, use the best solution
    if best_error > target_error and rank == 0:
        print(f"Warning: Target accuracy not reached. Best error = {best_error:.6e}")
    
    # Use the best solution found
    final_u = best_solution
    final_domain = best_domain
    final_V = best_V
    
    # Sample on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points)
    
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
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all points across processes
    u_values_all = np.zeros(points.shape[0]) if rank == 0 else None
    comm.Gather(u_values, u_values_all, root=0)
    
    if rank == 0:
        u_grid = u_values_all.reshape(ny, nx)
    else:
        u_grid = np.zeros((ny, nx))
    
    # Initial condition grid
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    u0_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_values_all = np.zeros(points.shape[0]) if rank == 0 else None
    comm.Gather(u0_values, u0_values_all, root=0)
    
    if rank == 0:
        u0_grid = u0_values_all.reshape(ny, nx)
    else:
        u0_grid = np.zeros((ny, nx))
    
    # Broadcast solver_info to all processes
    solver_info = comm.bcast(best_info, root=0)
    
    if rank == 0:
        print(f"Best domain L2 error achieved: {best_error:.6e}")
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test with a simple case specification
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
    if MPI.COMM_WORLD.rank == 0:
        print("Solution shape:", result["u"].shape)
        print("Solver info:", result["solver_info"])

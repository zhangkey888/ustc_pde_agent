import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive refinement within time budget.
    """
    start_time = time.time()
    time_limit = 39.683  # Maximum allowed time
    
    # Parameters
    t_end = 0.08
    dt_suggested = 0.008
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Manufactured solution
    def u_exact(x, t):
        return np.exp(-t) * np.exp(5*x[0]) * np.sin(np.pi*x[1])
    
    def f_source(x, t):
        u = np.exp(-t) * np.exp(5*x[0]) * np.sin(np.pi*x[1])
        du_dt = -u
        laplacian_u = (25 - np.pi**2) * u
        return du_dt - laplacian_u
    
    # Configurations to try (increasing cost)
    configs = [
        {"N": 64, "degree": 1, "dt_factor": 1.0},
        {"N": 128, "degree": 1, "dt_factor": 0.5},
        {"N": 192, "degree": 1, "dt_factor": 0.5},
        {"N": 256, "degree": 1, "dt_factor": 0.25},
    ]
    
    best_error = float('inf')
    best_result = None
    best_info = None
    
    for config in configs:
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.8:  # Use 80% of time limit
            print(f"Time budget used: {elapsed:.1f}s, using best result so far")
            break
        
        N = config["N"]
        degree = config["degree"]
        dt_factor = config["dt_factor"]
        
        # Create mesh
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Boundary conditions
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
        u_bc = fem.Function(V)
        
        # Time-stepping
        u_n = fem.Function(V)
        u = fem.Function(V)
        u_n.interpolate(lambda x: u_exact(x, 0.0))
        u.x.array[:] = u_n.x.array
        
        v = ufl.TestFunction(V)
        dt_value = dt_suggested * dt_factor
        dt = fem.Constant(domain, ScalarType(dt_value))
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        n_steps = int(np.ceil(t_end / dt_value))
        actual_dt = t_end / n_steps
        dt.value = actual_dt
        
        total_iterations = 0
        ksp_type = "gmres"
        pc_type = "hypre"
        rtol = 1e-8
        
        for step in range(n_steps):
            t = (step + 1) * actual_dt
            
            u_bc.interpolate(lambda x: u_exact(x, t))
            bc = fem.dirichletbc(u_bc, dofs)
            
            t_mid = t - actual_dt/2
            f_func = fem.Function(V)
            f_func.interpolate(lambda x: f_source(x, t_mid))
            
            u_trial = ufl.TrialFunction(V)
            a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
            
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
                    petsc_options_prefix="heat_"
                )
                u_sol = problem.solve()
                total_iterations += problem.solver.getIterationNumber()
            except:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                    petsc_options_prefix="heat_"
                )
                u_sol = problem.solve()
                total_iterations += problem.solver.getIterationNumber()
                ksp_type = "preonly"
                pc_type = "lu"
                rtol = 1e-12
            
            u.x.array[:] = u_sol.x.array
            u_n.x.array[:] = u.x.array
        
        # Sample on grid for error estimation
        nx, ny = 50, 50
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        points[2, :] = 0.0
        
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        
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
            vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape(nx, ny)
        
        # Compute exact solution on grid
        u_exact_val = u_exact(np.array([X, Y]), t_end)
        error_mask = ~np.isnan(u_grid)
        if np.any(error_mask):
            rms_error = np.sqrt(np.mean((u_grid[error_mask] - u_exact_val[error_mask])**2))
        else:
            rms_error = float('inf')
        
        # Store if best
        if rms_error < best_error:
            best_error = rms_error
            best_result = (u, domain, V, u_grid)
            best_info = {
                "mesh_resolution": N,
                "element_degree": degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": total_iterations,
                "dt": float(actual_dt),
                "n_steps": n_steps,
                "time_scheme": scheme
            }
        
        print(f"N={N}, degree={degree}, dt={actual_dt:.4f}: error={rms_error:.2e}")
        
        # Early exit if we meet accuracy (assuming 1.06e-03 is for RMS error)
        if rms_error <= 1.06e-03:
            print(f"Accuracy requirement met!")
            break
    
    # Use best result
    u_final, final_domain, final_V, u_grid = best_result
    solver_info = best_info
    
    # Get initial condition
    u0_func = fem.Function(final_V)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
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
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape(nx, ny)
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s, Best error: {best_error:.2e}")
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.008,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print(f"\nFinal configuration:")
    print(f"  Mesh: {result['solver_info']['mesh_resolution']}")
    print(f"  Degree: {result['solver_info']['element_degree']}")
    print(f"  Steps: {result['solver_info']['n_steps']}")
    print(f"  dt: {result['solver_info']['dt']:.6f}")

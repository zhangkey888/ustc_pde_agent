import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Parameters
    t_end = 0.08
    dt_suggested = 0.008
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Exact solution and source
    def exact_solution(x, t):
        return np.exp(-t) * np.exp(5*x[1]) * np.sin(np.pi*x[0])
    
    def source_term(x, t):
        pi = np.pi
        return np.exp(-t) * np.exp(5*x[1]) * np.sin(pi*x[0]) * (pi**2 - 26)
    
    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    element_degree = 1
    
    prev_norm = None
    u_final = None
    final_resolution = None
    solver_info_base = {}
    
    comm = MPI.COMM_WORLD
    
    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Time stepping parameters
        dt = dt_suggested
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps
        
        # Functions
        u_n = fem.Function(V)
        u_n1 = fem.Function(V)
        
        # Initial condition
        def u0_expr(x):
            return exact_solution(x, 0.0)
        u_n.interpolate(u0_expr)
        
        # Boundary facets
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
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # BC function
        u_bc = fem.Function(V)
        
        # Source function
        f = fem.Function(V)
        
        # Time stepping
        current_time = 0.0
        linear_iterations = 0
        
        for step in range(n_steps):
            current_time += dt
            
            # Update BC
            def bc_expr(x):
                return exact_solution(x, current_time)
            u_bc.interpolate(bc_expr)
            bc = fem.dirichletbc(u_bc, boundary_dofs)
            
            # Update source
            def f_expr(x):
                return source_term(x, current_time)
            f.interpolate(f_expr)
            
            # Variational form for this time step
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            κ = fem.Constant(domain, ScalarType(1.0))
            a = ufl.inner(u, v) * ufl.dx + dt * κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
            
            # Solve using LinearProblem (more robust)
            # Try iterative solver first, then direct
            solver_success = False
            for solver_type in ['iterative', 'direct']:
                try:
                    if solver_type == 'iterative':
                        petsc_options = {
                            "ksp_type": "gmres",
                            "pc_type": "hypre",
                            "ksp_rtol": 1e-8,
                            "ksp_atol": 1e-10,
                            "ksp_max_it": 1000
                        }
                    else:
                        petsc_options = {
                            "ksp_type": "preonly",
                            "pc_type": "lu"
                        }
                    
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options=petsc_options,
                        petsc_options_prefix=f"step{step}_"
                    )
                    u_n1 = problem.solve()
                    
                    # Get iteration count
                    ksp = problem._solver
                    its = ksp.getIterationNumber()
                    linear_iterations += its
                    
                    solver_success = True
                    break
                    
                except Exception as e:
                    if solver_type == 'iterative':
                        print(f"Iterative solver failed at step {step}: {e}. Trying direct...")
                        continue
                    else:
                        raise RuntimeError(f"Both solvers failed at step {step}: {e}")
            
            if not solver_success:
                raise RuntimeError(f"Failed to solve at step {step}")
            
            # Update for next step
            u_n.x.array[:] = u_n1.x.array
        
        # Compute norm at final time
        norm_form = fem.form(ufl.inner(u_n1, u_n1) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:
                u_final = u_n1
                final_resolution = N
                solver_info_base = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "rtol": 1e-8,
                    "iterations": linear_iterations,
                    "dt": dt,
                    "n_steps": n_steps,
                    "time_scheme": scheme
                }
                break
        
        prev_norm = norm_value
        u_final = u_n1
        final_resolution = N
        solver_info_base = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": scheme
        }
    
    # If loop completes without convergence
    if final_resolution is None:
        final_resolution = resolutions[-1]
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
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
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape((nx, ny))
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    solver_info = solver_info_base.copy()
    solver_info["wall_time_sec"] = wall_time
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    test_case = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.008,
                "scheme": "backward_euler"
            }
        }
    }
    
    try:
        result = solve(test_case)
        print("Solver executed successfully!")
        print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Wall time: {result['solver_info']['wall_time_sec']:.3f}s")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

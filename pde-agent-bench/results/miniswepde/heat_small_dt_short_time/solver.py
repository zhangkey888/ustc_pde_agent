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
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    Returns solution sampled on 50x50 grid.
    """
    start_time = time.time()
    
    # Parameters from problem description (override with case_spec if present)
    t_end = 0.06
    dt = 0.003
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        scheme = time_params.get('scheme', scheme)
    
    # Exact manufactured solution
    def exact_solution(x, t):
        """u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)"""
        return np.exp(-t) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
    
    def source_term(x, t):
        """f = ∂u/∂t - ∇·(κ ∇u) with κ=1.0"""
        u_val = exact_solution(x, t)
        return u_val * (32*np.pi**2 - 1)
    
    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    comm = MPI.COMM_WORLD
    element_degree = 1  # Linear elements
    
    u_final = None
    mesh_resolution_used = None
    prev_norm = None
    converged = False
    
    # Solver info to collect
    solver_info = {
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": 0,
        "dt": dt,
        "n_steps": int(t_end / dt),
        "time_scheme": scheme,
    }
    
    total_linear_iterations = 0
    
    for N in resolutions:
        if converged:
            break
            
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Mark all boundaries for Dirichlet BC
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
        
        # Initial condition
        u_n = fem.Function(V)
        def u0_expr(x):
            return exact_solution(x, 0.0)
        u_n.interpolate(u0_expr)
        
        # Time-stepping
        t = 0.0
        n_steps = int(t_end / dt)
        if n_steps == 0:
            n_steps = 1
        
        # Try iterative solver first, fallback to direct if needed
        solver_success = False
        linear_iterations_this_resolution = 0
        ksp_type_used = "gmres"
        pc_type_used = "hypre"
        
        for solver_config in [("gmres", "hypre"), ("preonly", "lu")]:
            if solver_success:
                break
                
            try:
                # Reset to initial condition
                u_n.interpolate(u0_expr)
                t = 0.0
                
                # Time-stepping loop
                for step in range(n_steps):
                    t += dt
                    
                    # Boundary condition at current time
                    u_bc = fem.Function(V)
                    def bc_expr(x):
                        return exact_solution(x, t)
                    u_bc.interpolate(bc_expr)
                    
                    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
                    bc = fem.dirichletbc(u_bc, boundary_dofs)
                    
                    # Source term at current time
                    f = fem.Function(V)
                    def f_expr(x):
                        return source_term(x, t)
                    f.interpolate(f_expr)
                    
                    # Backward Euler variational form
                    v = ufl.TestFunction(V)
                    u = ufl.TrialFunction(V)
                    κ = fem.Constant(domain, ScalarType(1.0))
                    
                    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
                    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
                    
                    # Solve using LinearProblem
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={
                            "ksp_type": solver_config[0],
                            "pc_type": solver_config[1],
                            "ksp_rtol": 1e-8,
                            "ksp_max_it": 1000,
                            "ksp_error_if_not_converged": True
                        },
                        petsc_options_prefix="heat_"
                    )
                    
                    u_sol = problem.solve()
                    
                    # Get iteration count from the solver
                    ksp = problem.solver
                    its = ksp.getIterationNumber()
                    linear_iterations_this_resolution += its
                    
                    # Update for next time step
                    u_n.x.array[:] = u_sol.x.array
                
                solver_success = True
                ksp_type_used = solver_config[0]
                pc_type_used = solver_config[1]
                
            except PETSc.Error as e:
                # PETSc solver error, try direct solver if this was iterative
                if solver_config[0] == "preonly":
                    # Direct solver also failed
                    raise RuntimeError(f"Direct solver failed: {e}")
                # Otherwise continue to try direct solver
                print(f"Iterative solver failed, trying direct solver: {e}")
            except Exception as e:
                if solver_config[0] == "preonly":
                    raise
                print(f"Iterative solver failed, trying direct solver: {e}")
        
        total_linear_iterations += linear_iterations_this_resolution
        
        # Final solution at t_end
        u_final_current = fem.Function(V)
        u_final_current.x.array[:] = u_n.x.array
        
        # Compute L2 norm for convergence check
        norm_form = fem.form(ufl.inner(u_final_current, u_final_current) * ufl.dx)
        norm = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence between mesh resolutions
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 0 else 0
            if relative_error < 0.01:  # 1% convergence criterion
                converged = True
                mesh_resolution_used = N
                u_final = u_final_current
                break
        
        prev_norm = norm
        mesh_resolution_used = N
        u_final = u_final_current
    
    # If loop finished without convergence, use finest mesh
    if not converged:
        mesh_resolution_used = resolutions[-1]
    
    # Update solver info with actual values
    solver_info.update({
        "mesh_resolution": mesh_resolution_used,
        "iterations": total_linear_iterations,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
    })
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create points in 3D format (required by dolfinx)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
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
    
    # Reshape to 50x50 grid
    u_grid = u_values.reshape((ny, nx))
    
    # Also get initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    u_initial = u0_values.reshape((ny, nx))
    
    # Check time limit
    elapsed = time.time() - start_time
    if elapsed > 27.472:
        print(f"Warning: Solve time {elapsed:.2f}s exceeds limit 27.472s")
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.06,
                "dt": 0.003,
                "scheme": "backward_euler"
            }
        }
    }
    result = solve(case_spec)
    print("Solver test completed successfully")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Total linear iterations: {result['solver_info']['iterations']}")
    print(f"u shape: {result['u'].shape}")
    
    # Quick accuracy check
    def exact_solution_test(x, y, t):
        return np.exp(-t) * np.sin(4*np.pi*x) * np.sin(4*np.pi*y)
    
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    u_exact = exact_solution_test(X, Y, 0.06)
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"Max error: {max_error:.2e}")
    if max_error <= 8.90e-03:
        print("✓ Accuracy requirement met")
    else:
        print(f"✗ Accuracy requirement NOT met")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and runtime auto-tuning.
    """
    # Extract parameters from case_spec with defaults
    pde_time = case_spec.get('pde', {}).get('time', {})
    # Hardcoded defaults as per problem description
    t_end = pde_time.get('t_end', 0.1)
    dt_suggested = pde_time.get('dt', 0.01)
    scheme = pde_time.get('scheme', 'backward_euler')
    
    # Force transient
    dt = float(dt_suggested)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # Adjust to exactly reach t_end
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    comm = MPI.COMM_WORLD
    u_solutions = []
    norms = []
    
    # Solver iterations counter
    total_linear_iterations = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define kappa as a function
        kappa = fem.Function(V)
        kappa.interpolate(lambda x: 0.2 + np.exp(-120 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2)))
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step
        
        # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
        u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Boundary conditions: Dirichlet from manufactured solution
        def boundary(x):
            return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)) | \
                   np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
        
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Source term function
        f_func = fem.Function(V)
        
        # Variational form for backward Euler (linear)
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
        
        # Time-stepping loop
        t_val = 0.0
        linear_iterations_this_resolution = 0
        solver_used = "direct"  # default
        
        for step in range(n_steps):
            t_val += dt
            # Update BC to exact solution at time t_val
            u_bc.interpolate(lambda x: np.exp(-t_val) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
            
            # Compute source term f at time t_val
            def compute_f(x, t):
                x0 = x[0]
                x1 = x[1]
                u_val = np.exp(-t) * np.sin(np.pi * x0) * np.sin(np.pi * x1)
                du_dt = -u_val
                # grad u
                grad_u_x = np.exp(-t) * np.pi * np.cos(np.pi * x0) * np.sin(np.pi * x1)
                grad_u_y = np.exp(-t) * np.pi * np.sin(np.pi * x0) * np.cos(np.pi * x1)
                # kappa
                kappa_val = 0.2 + np.exp(-120 * ((x0 - 0.55)**2 + (x1 - 0.45)**2))
                # grad kappa
                r2 = (x0 - 0.55)**2 + (x1 - 0.45)**2
                dkappa_dx = -240 * (x0 - 0.55) * np.exp(-120 * r2)
                dkappa_dy = -240 * (x1 - 0.45) * np.exp(-120 * r2)
                # div(kappa * grad u) = kappa * laplacian(u) + grad(kappa)·grad(u)
                laplacian_u = -2 * np.pi**2 * u_val
                div_term = kappa_val * laplacian_u + dkappa_dx * grad_u_x + dkappa_dy * grad_u_y
                f_val = du_dt - div_term
                return f_val
            
            f_func.interpolate(lambda x: compute_f(x, t_val))
            
            # Try iterative solver first, fallback to direct (only first step decides)
            if step == 0:
                try:
                    # Use iterative solver
                    problem = petsc.LinearProblem(a, L, bcs=[bc], 
                                                  petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                                                  petsc_options_prefix="heat_")
                    u_sol = problem.solve()
                    # Get iteration count
                    ksp = problem.solver
                    its = ksp.getIterationNumber()
                    linear_iterations_this_resolution += its
                    solver_used = "iterative"
                except Exception as e:
                    # Fallback to direct solver
                    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                                  petsc_options_prefix="heat_")
                    u_sol = problem.solve()
                    ksp = problem.solver
                    its = ksp.getIterationNumber()
                    linear_iterations_this_resolution += its
                    solver_used = "direct"
            else:
                # Reuse the same solver type
                if solver_used == "iterative":
                    problem = petsc.LinearProblem(a, L, bcs=[bc], 
                                                  petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                                                  petsc_options_prefix="heat_")
                else:
                    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                                  petsc_options_prefix="heat_")
                u_sol = problem.solve()
                ksp = problem.solver
                its = ksp.getIterationNumber()
                linear_iterations_this_resolution += its
            
            # Update solution
            u.x.array[:] = u_sol.x.array[:]
            # Update previous solution
            u_n.x.array[:] = u.x.array[:]
        
        # Compute norm of solution
        norm = fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx))
        norms.append(norm)
        u_solutions.append((u, domain, V, solver_used))
        total_linear_iterations += linear_iterations_this_resolution
        
        # Check convergence
        if len(norms) >= 2:
            rel_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] != 0 else 1.0
            if rel_error < 0.01:
                break
    
    # Use the last solution
    u_final, domain_final, V_final, solver_used = u_solutions[-1]
    final_N = resolutions[len(u_solutions)-1]
    
    # Sample on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T
    
    # Evaluate solution at points
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points)
    
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
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny, nx)
    
    # Initial condition grid
    u0_func = fem.Function(V_final)
    u0_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u0_vals = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_vals[eval_map] = vals0.flatten()
    u_initial = u0_vals.reshape(ny, nx)
    
    # Determine solver types for info
    if solver_used == "iterative":
        ksp_type = "gmres"
        pc_type = "hypre"
    else:
        ksp_type = "preonly"
        pc_type = "lu"
    
    # Solver info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": 1,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test with a dummy case_spec
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
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

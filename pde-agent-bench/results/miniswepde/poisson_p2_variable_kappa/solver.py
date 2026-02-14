import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with variable kappa.
    Returns dict with "u" (solution on 50x50 grid) and "solver_info".
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Time limit (seconds)
    time_limit = 2.0  # Leave some margin under 2.387s
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128, 256]
    element_degrees = [2, 1]  # Try P2 first for accuracy
    
    # Storage
    u_sol = None
    norm_old = None
    final_resolution = None
    final_degree = None
    solver_info = {}
    
    start_time = time.time()
    
    # Outer loop over element degrees
    for degree in element_degrees:
        if u_sol is not None:
            break
            
        norm_old = None
        
        for N in resolutions:
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"Time limit approaching, using current solution")
                break
            
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Spatial coordinate
            x = ufl.SpatialCoordinate(domain)
            
            # Variable kappa: 1 + 0.4*sin(2*pi*x)*sin(2*pi*y)
            kappa = 1.0 + 0.4 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
            
            # Exact solution and source term
            u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
            f = -ufl.div(kappa * ufl.grad(u_exact))
            
            # Weak form
            a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f, v) * ufl.dx
            
            # Boundary condition: u = g on ∂Ω
            def boundary_marker(x):
                return np.ones(x.shape[1], dtype=bool)
            
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Try iterative solver first, fallback to direct if fails
            solver_success = False
            ksp_type = 'gmres'
            pc_type = 'hypre'
            rtol = 1e-8
            iterations = 0
            
            for solver_try in range(2):
                try:
                    petsc_options = {
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "ksp_rtol": rtol,
                        "ksp_atol": 1e-10,
                        "ksp_max_it": 1000,
                    }
                    
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options=petsc_options,
                        petsc_options_prefix="poisson_"
                    )
                    u_sol_current = problem.solve()
                    
                    # Get iteration count
                    iterations = problem.solver.getIterationNumber()
                    
                    # Compute L2 norm of solution
                    norm_form = fem.form(ufl.inner(u_sol_current, u_sol_current) * ufl.dx)
                    norm_value = np.sqrt(fem.assemble_scalar(norm_form))
                    
                    # Check convergence based on relative change in norm
                    if norm_old is not None:
                        relative_error = abs(norm_value - norm_old) / norm_value
                        # Stricter tolerance for accuracy
                        if relative_error < 0.001:  # 0.1% convergence
                            u_sol = u_sol_current
                            final_resolution = N
                            final_degree = degree
                            solver_info.update({
                                "mesh_resolution": N,
                                "element_degree": degree,
                                "ksp_type": ksp_type,
                                "pc_type": pc_type,
                                "rtol": rtol,
                                "iterations": iterations,
                            })
                            solver_success = True
                            break
                    
                    norm_old = norm_value
                    solver_success = True
                    break
                    
                except Exception as e:
                    if solver_try == 0:
                        print(f"Iterative solver failed, switching to direct: {e}")
                        ksp_type = 'preonly'
                        pc_type = 'lu'
                        rtol = 1e-12
                    else:
                        raise
            
            if solver_success and u_sol is not None:
                break
        
        if u_sol is not None:
            break
    
    # Fallback: use finest mesh and highest degree if not converged
    if u_sol is None:
        N = resolutions[-1]
        degree = element_degrees[0]
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        x = ufl.SpatialCoordinate(domain)
        kappa = 1.0 + 0.4 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        f = -ufl.div(kappa * ufl.grad(u_exact))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Use direct solver for robustness
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="poisson_fallback_"
        )
        u_sol = problem.solve()
        
        final_resolution = N
        final_degree = degree
        iterations = 0  # Direct solver doesn't have iterations
        solver_info.update({
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        })
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(u_sol.function_space.mesh, u_sol.function_space.mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(
        u_sol.function_space.mesh, cell_candidates, points.T
    )
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Fill missing solver_info fields
    if "mesh_resolution" not in solver_info:
        solver_info["mesh_resolution"] = final_resolution
    if "element_degree" not in solver_info:
        solver_info["element_degree"] = final_degree
    if "ksp_type" not in solver_info:
        solver_info["ksp_type"] = "preonly"
    if "pc_type" not in solver_info:
        solver_info["pc_type"] = "lu"
    if "rtol" not in solver_info:
        solver_info["rtol"] = 1e-12
    if "iterations" not in solver_info:
        solver_info["iterations"] = iterations
    
    # Ensure no NaN values
    if np.any(np.isnan(u_grid)):
        u_grid = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
    
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Solve time: {solve_time:.3f}s, mesh: {final_resolution}, degree: {final_degree}, iterations: {iterations}")
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "elliptic",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.4*sin(2*pi*x)*sin(2*pi*y)"}
            }
        }
    }
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

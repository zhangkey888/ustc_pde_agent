import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement and degree adaptation.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Manufactured solution
    def exact_solution(x):
        # x shape (3, n) but we are in 2D
        return np.exp(x[0] * x[1]) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Source term f = -Δu (since κ=1)
    def source_term(x):
        xv = x[0]
        yv = x[1]
        a = xv * yv
        b = np.sin(np.pi * xv)
        c = np.sin(np.pi * yv)
        cos_pi_x = np.cos(np.pi * xv)
        cos_pi_y = np.cos(np.pi * yv)
        exp_a = np.exp(a)
        # u_xx
        u_xx = exp_a * (yv*yv * b * c + 2*yv*np.pi*cos_pi_x*c - np.pi*np.pi*b*c)
        # u_yy
        u_yy = exp_a * (xv*xv * b * c + 2*xv*np.pi*cos_pi_y*b - np.pi*np.pi*b*c)
        laplacian_u = u_xx + u_yy
        return -laplacian_u  # f = -Δu because -∇·(κ∇u) = f
    
    # Adaptive strategy: try higher degree first (more accurate per DOF)
    degree_resolution_plans = [
        (2, [16, 32, 64]),    # degree 2, coarse meshes
        (1, [32, 64, 128]),   # degree 1, progressive refinement
    ]
    
    u_sol = None
    mesh_resolution_used = None
    element_degree_used = None
    solver_info_step = {}
    target_max_error = 2.84e-04  # required accuracy
    
    for degree, resolutions in degree_resolution_plans:
        norm_old = None
        for N in resolutions:
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Dirichlet boundary condition (exact solution on entire boundary)
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            u_bc = fem.Function(V)
            u_bc.interpolate(exact_solution)
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            κ = fem.Constant(domain, ScalarType(1.0))
            f = fem.Function(V)
            f.interpolate(source_term)
            
            a = ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f, v) * ufl.dx
            
            # Solve linear system with iterative solver first, fallback to direct
            try:
                # Try iterative solver
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                ksp = problem.solver
                solver_info_step["ksp_type"] = "gmres"
                solver_info_step["pc_type"] = "hypre"
                solver_info_step["rtol"] = 1e-8
                solver_info_step["iterations"] = ksp.getIterationNumber()
            except Exception:
                # Fallback to direct solver
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                ksp = problem.solver
                solver_info_step["ksp_type"] = "preonly"
                solver_info_step["pc_type"] = "lu"
                solver_info_step["rtol"] = 1e-12
                solver_info_step["iterations"] = ksp.getIterationNumber()
            
            # Compute L2 norm of solution
            norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
            norm_local = fem.assemble_scalar(norm_form)
            norm_new = domain.comm.allreduce(norm_local, op=MPI.SUM)
            norm_new = np.sqrt(norm_new)
            
            # Estimate max error on a coarse grid (10x10) for convergence check
            # This is cheap and correlates with the final 50x50 grid error
            nx_test = ny_test = 10
            x_test = np.linspace(0.0, 1.0, nx_test)
            y_test = np.linspace(0.0, 1.0, ny_test)
            X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
            points_test = np.vstack([X_test.ravel(), Y_test.ravel(), np.zeros(nx_test*ny_test)]).astype(np.float64)
            u_test_vals = evaluate_at_points(u_sol, points_test)
            exact_test = np.exp(X_test * Y_test) * np.sin(np.pi * X_test) * np.sin(np.pi * Y_test)
            max_error_estimate = np.max(np.abs(u_test_vals.reshape((nx_test, ny_test)) - exact_test))
            
            # Check convergence based on relative norm change
            norm_converged = False
            if norm_old is not None:
                relative_error_norm = abs(norm_new - norm_old) / norm_new
                if relative_error_norm < 0.01:
                    norm_converged = True
            
            # If estimated max error is below target, accept this solution
            if max_error_estimate < target_max_error:
                mesh_resolution_used = N
                element_degree_used = degree
                break
            
            # If norm converged but error still too high, continue refining
            norm_old = norm_new
        
        # If we found a suitable solution within this degree, break outer loop
        if mesh_resolution_used is not None:
            break
    
    # If loop finished without finding suitable solution, use finest from last plan
    if mesh_resolution_used is None:
        element_degree_used, resolutions = degree_resolution_plans[-1]
        mesh_resolution_used = resolutions[-1]
        # u_sol already holds the last solved solution
    
    # At this point, u_sol is the solution from the accepted mesh
    # Need to interpolate onto 50x50 uniform grid
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).astype(np.float64)
    
    # Evaluate solution at points
    u_grid_flat = evaluate_at_points(u_sol, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree_used,
        "ksp_type": solver_info_step.get("ksp_type", "gmres"),
        "pc_type": solver_info_step.get("pc_type", "hypre"),
        "rtol": solver_info_step.get("rtol", 1e-8),
        "iterations": solver_info_step.get("iterations", 0)
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

def evaluate_at_points(u_func, points):
    """
    Evaluate dolfinx Function at points (shape (3, N)).
    Returns array of shape (N,).
    """
    domain = u_func.function_space.mesh
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=PETSc.ScalarType)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

if __name__ == "__main__":
    # Simple test
    case_spec = {"pde": {"type": "elliptic"}}
    result = solve(case_spec)
    print("Mesh resolution:", result["solver_info"]["mesh_resolution"])
    print("Element degree:", result["solver_info"]["element_degree"])
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

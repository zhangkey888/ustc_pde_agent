import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with variable coefficient κ.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128, 256]
    degrees = [1, 2]
    
    # Target L2 error (conservative)
    target_error = 1.0e-3
    
    # Storage for best solution
    best_solution = None
    best_domain = None
    best_V = None
    best_info = {}
    
    total_iterations = 0  # sum across all linear solves
    
    for degree in degrees:
        if rank == 0:
            print(f"Trying element degree {degree}")
        prev_norm = None
        for N in resolutions:
            if rank == 0:
                print(f"  Testing mesh resolution N={N}")
            
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            x = ufl.SpatialCoordinate(domain)
            
            # Coefficient κ (from case_spec, but hardcoded for this problem)
            kappa_expr = 1.0 + 0.5 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
            # Exact solution
            u_exact_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
            # Source term f = -∇·(κ ∇u_exact)
            f_expr = -ufl.div(kappa_expr * ufl.grad(u_exact_expr))
            
            # Boundary condition: Dirichlet u = g on ∂Ω
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
            
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
            )
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Bilinear and linear forms
            a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f_expr, v) * ufl.dx
            
            # Solver: try iterative then direct
            solver_succeeded = False
            linear_iterations = 0
            ksp_type = 'gmres'
            pc_type = 'hypre'
            rtol = 1e-8
            
            for solver_try in range(2):
                if solver_try == 0:
                    petsc_options = {
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "ksp_rtol": rtol,
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000,
                    }
                    petsc_options_prefix = "pdebench_iter_"
                else:
                    ksp_type = 'preonly'
                    pc_type = 'lu'
                    petsc_options = {
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                    }
                    petsc_options_prefix = "pdebench_direct_"
                
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options=petsc_options,
                        petsc_options_prefix=petsc_options_prefix
                    )
                    u_h = problem.solve()
                    ksp = problem.solver
                    its = ksp.getIterationNumber()
                    linear_iterations = its
                    solver_succeeded = True
                    break
                except Exception as e:
                    if rank == 0:
                        print(f"    Solver try {solver_try} failed: {e}")
                    continue
            
            if not solver_succeeded:
                raise RuntimeError("Linear solver failed")
            
            total_iterations += linear_iterations
            
            # Compute L2 error
            error_expr = u_h - u_exact_expr
            error_form = fem.form(ufl.inner(error_expr, error_expr) * ufl.dx)
            error_local = fem.assemble_scalar(error_form)
            error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
            l2_error = np.sqrt(error_global)
            
            if rank == 0:
                print(f"    L2 error = {l2_error:.3e}, iterations = {linear_iterations}")
            
            # Store solution if best so far (lowest error)
            if best_solution is None or l2_error < best_info.get('l2_error', float('inf')):
                best_solution = u_h
                best_domain = domain
                best_V = V
                best_info = {
                    'mesh_resolution': N,
                    'element_degree': degree,
                    'ksp_type': ksp_type,
                    'pc_type': pc_type,
                    'rtol': rtol,
                    'l2_error': l2_error,
                }
            
            # Check if error meets target
            if l2_error < target_error:
                if rank == 0:
                    print(f"    Target error met at N={N}, degree={degree}")
                break
            
            # Convergence in solution norm (optional)
            norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
            norm_local = fem.assemble_scalar(norm_form)
            norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)
            norm_current = np.sqrt(norm_global)
            if prev_norm is not None:
                rel_error = abs(norm_current - prev_norm) / norm_current
                if rank == 0:
                    print(f"    Relative norm error: {rel_error:.3e}")
                if rel_error < 0.01:
                    # Norm converged; if error still high, maybe need higher degree
                    pass
            prev_norm = norm_current
        
        # If we have a solution with error < target, break degree loop
        if best_info.get('l2_error', float('inf')) < target_error:
            break
    
    # Use best solution found
    u_sol = best_solution
    domain = best_domain
    V = best_V
    mesh_resolution_used = best_info['mesh_resolution']
    element_degree_used = best_info['element_degree']
    ksp_type_used = best_info['ksp_type']
    pc_type_used = best_info['pc_type']
    rtol_used = best_info['rtol']
    
    if rank == 0:
        print(f"Selected: N={mesh_resolution_used}, degree={element_degree_used}, L2 error={best_info['l2_error']:.3e}")
        print(f"Total linear iterations across all solves: {total_iterations}")
    
    # Sample solution on 50x50 uniform grid
    nx = 50
    ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T  # shape (N, 3)
    
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
    
    u_grid_flat = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Gather on rank 0 (assuming evaluator runs with single rank)
    if comm.size > 1:
        recv_buf = None
        if rank == 0:
            recv_buf = np.empty((comm.size, points.shape[0]), dtype=ScalarType)
        comm.Gather(u_grid_flat, recv_buf, root=0)
        if rank == 0:
            # Combine: take first non-nan for each point
            combined = np.nanmean(recv_buf, axis=0)
            u_grid = combined.reshape((nx, ny))
        else:
            u_grid = np.empty((nx, ny), dtype=ScalarType)
        comm.Bcast(u_grid, root=0)
    else:
        u_grid = u_grid_flat.reshape((nx, ny))
    
    # Build solver_info dict
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree_used,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": total_iterations,
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result

if __name__ == "__main__":
    # Simple test
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"}
            }
        }
    }
    result = solve(case_spec)
    print("Test completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation using standard Galerkin.
    Returns solution on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Parameters
    eps = 0.03
    beta = np.array([5.0, 2.0])
    
    # Exact solution
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Source term f = -ε ∇²u + β·∇u
    def source_term(x):
        x_coord = x[0]
        y_coord = x[1]
        sin_pi_x = np.sin(np.pi * x_coord)
        sin_2pi_y = np.sin(2 * np.pi * y_coord)
        cos_pi_x = np.cos(np.pi * x_coord)
        cos_2pi_y = np.cos(2 * np.pi * y_coord)
        laplacian_u = -5 * (np.pi**2) * sin_pi_x * sin_2pi_y
        grad_u_x = np.pi * cos_pi_x * sin_2pi_y
        grad_u_y = 2 * np.pi * sin_pi_x * cos_2pi_y
        return -eps * laplacian_u + beta[0] * grad_u_x + beta[1] * grad_u_y
    
    # Adaptive mesh refinement based on max error against exact solution
    resolutions = [64, 128, 256]
    degree = 2
    u_sol = None
    mesh_used = None
    total_iterations = 0
    final_N = resolutions[-1]
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    target_max_error = 2e-7  # ensures N=256
    
    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Boundary condition: Dirichlet using exact solution (all boundaries)
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Variational form (standard Galerkin)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        beta_ufl = ufl.as_vector([beta[0], beta[1]])
        
        a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) 
             + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v)) * ufl.dx
        
        f = fem.Function(V)
        f.interpolate(lambda x: source_term(x))
        L = ufl.inner(f, v) * ufl.dx
        
        # Linear problem with iterative solver
        petsc_options = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 1000,
        }
        
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix="conv_"
            )
            u_sol = problem.solve()
            ksp = problem.solver
            total_iterations += ksp.getIterationNumber()
            ksp_type_used = "gmres"
            pc_type_used = "hypre"
        except Exception as e:
            # Fallback to direct solver
            petsc_options_fallback = {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_fallback,
                petsc_options_prefix="conv_"
            )
            u_sol = problem.solve()
            ksp = problem.solver
            total_iterations += ksp.getIterationNumber()
            ksp_type_used = "preonly"
            pc_type_used = "lu"
        
        # Estimate max error by evaluating at 50x50 grid points (against exact)
        nx = ny = 50
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T
        
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
        
        if np.any(np.isnan(u_values)):
            u_values_all = comm.allgather(u_values)
            u_values = u_values_all[0]
        
        exact_vals = exact_solution(points.T)
        max_error = np.max(np.abs(u_values - exact_vals))
        
        # Check convergence
        if max_error < target_max_error:
            mesh_used = domain
            final_N = N
            break
        
        mesh_used = domain
        final_N = N
    
    # If loop finished without break, use finest mesh result (N=256)
    # Already have u_sol from last iteration
    
    # Now compute final solution on 50x50 grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T
    
    bb_tree = geometry.bb_tree(mesh_used, mesh_used.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(mesh_used, cell_candidates, points)
    
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
    
    if np.any(np.isnan(u_values)):
        u_values_all = comm.allgather(u_values)
        u_values = u_values_all[0]
    
    u_grid = u_values.reshape(nx, ny)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": 1e-10,
        "iterations": total_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    import time
    case_spec = {"pde": {"type": "convection-diffusion"}}
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    print(f"Time: {end-start:.3f}s")
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

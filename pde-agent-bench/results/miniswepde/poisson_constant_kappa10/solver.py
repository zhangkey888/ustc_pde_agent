import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Problem parameters
    kappa = 10.0
    exact_solution = lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Manufactured source term f = -∇·(κ ∇u_exact)
    # u_exact = sin(pi*x)*sin(2*pi*y)
    # ∇u = [pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y)]
    # Δu = -pi^2*sin(pi*x)*sin(2*pi*y) - 4*pi^2*sin(pi*x)*sin(2*pi*y) 
    #     = -5*pi^2*sin(pi*x)*sin(2*pi*y)
    # So f = -κ*Δu = κ*5*pi^2*sin(pi*x)*sin(2*pi*y)
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    element_degree = 1  # P1 elements
    
    # Solver parameters (try iterative first, fallback to direct)
    solver_params = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "max_it": 1000
    }
    
    # Storage for convergence check
    prev_norm = None
    u_sol_final = None
    domain_final = None
    mesh_resolution_used = None
    iterations_total = 0
    solver_type_used = "iterative"
    ksp_type_used = solver_params["ksp_type"]
    pc_type_used = solver_params["pc_type"]
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # All boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create BC function
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term f = κ*5*pi^2*sin(pi*x)*sin(2*pi*y)
        x = ufl.SpatialCoordinate(domain)
        f_expr = kappa * 5.0 * (np.pi**2) * ufl.sin(np.pi * x[0]) * ufl.sin(2.0 * np.pi * x[1])
        
        # Create function for source term
        f_function = fem.Function(V)
        f_expr_compiled = fem.Expression(f_expr, V.element.interpolation_points)
        f_function.interpolate(f_expr_compiled)
        
        # Weak form: ∫(κ ∇u·∇v) dx = ∫ f v dx
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_function, v) * ufl.dx
        
        # Try iterative solver first
        u_sol = fem.Function(V)
        iterations = 0
        success = False
        
        try:
            # Create linear problem with iterative solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": solver_params["ksp_type"],
                    "pc_type": solver_params["pc_type"],
                    "ksp_rtol": solver_params["rtol"],
                    "ksp_max_it": solver_params["max_it"]
                },
                petsc_options_prefix="poisson_"
            )
            problem.solve()
            
            # Get iteration count
            ksp = problem.solver
            iterations = ksp.getIterationNumber()
            iterations_total += iterations
            solver_type_used = "iterative"
            ksp_type_used = solver_params["ksp_type"]
            pc_type_used = solver_params["pc_type"]
            success = True
            
            if rank == 0:
                print(f"  Iterative solver converged in {iterations} iterations")
                
        except Exception as e:
            if rank == 0:
                print(f"  Iterative solver failed: {e}, trying direct solver")
        
        if not success:
            # Fallback to direct solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                },
                petsc_options_prefix="poisson_"
            )
            problem.solve()
            
            # Direct solver doesn't have iteration count in the same way
            iterations = 0
            solver_type_used = "direct"
            ksp_type_used = "preonly"
            pc_type_used = "lu"
            
            if rank == 0:
                print(f"  Direct solver succeeded")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        if rank == 0:
            print(f"  L2 norm: {norm_value:.6f}")
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value
            if rank == 0:
                print(f"  Relative error: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Converged at N={N}")
                u_sol_final = u_sol
                domain_final = domain
                mesh_resolution_used = N
                break
        
        prev_norm = norm_value
        u_sol_final = u_sol
        domain_final = domain
        mesh_resolution_used = N
    
    # If loop finished without break, use the last solution (N=128)
    if u_sol_final is None:
        u_sol_final = u_sol
        domain_final = domain
        mesh_resolution_used = 128
    
    # Interpolate solution to 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array with shape (3, nx*ny)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    u_grid_flat = np.full((nx * ny,), np.nan, dtype=ScalarType)
    
    # Only rank 0 needs to evaluate for output
    if rank == 0:
        # Create bounding box tree
        bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
        
        # Find cells containing points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points.T)
        
        # Build lists of points and cells for evaluation
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        if len(points_on_proc) > 0:
            vals = u_sol_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_grid_flat[eval_map] = vals.flatten()
    
    # Broadcast the result to all ranks (though only rank 0 has non-NaN values)
    comm.Bcast(u_grid_flat, root=0)
    
    # Reshape to (nx, ny)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": solver_params["rtol"],
        "iterations": iterations_total
    }
    
    # No time-dependent fields needed for elliptic problem
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    return result

# Test the solver if run directly
if __name__ == "__main__":
    # Create a minimal case_spec for testing
    case_spec = {
        "pde": {
            "type": "elliptic"
        }
    }
    
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Min/max of solution: {result['u'].min():.6f}, {result['u'].max():.6f}")
    print(f"Solver info: {result['solver_info']}")

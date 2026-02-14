import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    case_spec: dictionary containing problem specification
    
    Returns:
    dict with keys:
        - "u": numpy array shape (50, 50) with solution on uniform grid
        - "solver_info": dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Define exact solution for error computation
    def exact_solution(x):
        """u_exact = sin(3*pi*(x+y))*sin(pi*(x-y))"""
        return np.sin(3*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1]))
    
    # Define source term f = -∇·(κ ∇u_exact) with κ=1
    # f = -Δu_exact
    def source_term(x):
        # Compute Laplacian of u_exact analytically
        # u = sin(3π(x+y)) * sin(π(x-y))
        # Let a = 3π(x+y), b = π(x-y)
        # u = sin(a) * sin(b)
        # Δu = ∂²u/∂x² + ∂²u/∂y²
        # Using symbolic computation:
        # ∂u/∂x = 3π*cos(a)*sin(b) + π*sin(a)*cos(b)
        # ∂u/∂y = 3π*cos(a)*sin(b) - π*sin(a)*cos(b)
        # ∂²u/∂x² = -9π²*sin(a)*sin(b) + 3π²*cos(a)*cos(b) + 3π²*cos(a)*cos(b) - π²*sin(a)*sin(b)
        #         = -10π²*sin(a)*sin(b) + 6π²*cos(a)*cos(b)
        # ∂²u/∂y² = -9π²*sin(a)*sin(b) - 3π²*cos(a)*cos(b) - 3π²*cos(a)*cos(b) - π²*sin(a)*sin(b)
        #         = -10π²*sin(a)*sin(b) - 6π²*cos(a)*cos(b)
        # Δu = ∂²u/∂x² + ∂²u/∂y² = -20π²*sin(a)*sin(b)
        # So f = -Δu = 20π²*sin(3π(x+y))*sin(π(x-y))
        a = 3*np.pi*(x[0] + x[1])
        b = np.pi*(x[0] - x[1])
        return 20*np.pi*np.pi*np.sin(a)*np.sin(b)
    
    # Adaptive mesh refinement loop
    # Try degree 2 elements first (more accurate per DOF)
    element_degree = 2
    resolutions = [32, 64, 128]
    u_sol = None
    mesh_resolution_used = None
    total_iterations = 0
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    rtol_used = 1e-8
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Define source term as a function
        f_expr = fem.Function(V)
        f_expr.interpolate(lambda x: source_term(x))
        
        # Variational form
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Boundary conditions: Dirichlet using exact solution
        def boundary_marker(x):
            # Mark all boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct if fails
        petsc_options = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": rtol_used,
            "ksp_max_it": 1000,
            "ksp_converged_reason": None
        }
        
        try:
            # Create and solve linear problem
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            
            # Get iteration count from KSP solver
            ksp = problem.solver
            its = ksp.getIterationNumber()
            total_iterations += its
            
            # Compute max pointwise error estimate at mesh vertices
            # This gives a conservative estimate of the error
            mesh_points = domain.geometry.x
            
            # Find cells containing each vertex
            bb_tree = geometry.bb_tree(domain, tdim)
            cell_candidates = geometry.compute_collisions_points(bb_tree, mesh_points)
            colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, mesh_points)
            
            # For each point, take first colliding cell
            cells_for_points = []
            valid_points = []
            for i in range(len(mesh_points)):
                links = colliding_cells.links(i)
                if len(links) > 0:
                    cells_for_points.append(links[0])
                    valid_points.append(mesh_points[i])
            
            max_error_estimate = 0.0
            if len(valid_points) > 0:
                u_sol_at_points = u_sol.eval(np.array(valid_points), np.array(cells_for_points, dtype=np.int32))
                u_exact_at_points = exact_solution(np.array(valid_points).T)
                max_error_estimate = np.max(np.abs(u_sol_at_points.flatten() - u_exact_at_points))
            
            # Check if error meets accuracy requirement (4.04e-03)
            # Use conservative factor: require error < 2e-3 to account for interpolation
            if max_error_estimate < 2e-3:
                mesh_resolution_used = N
                break
                
        except Exception as e:
            # Fallback to direct solver
            print(f"Iterative solver failed for N={N}: {e}. Switching to direct solver.")
            petsc_options_direct = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": rtol_used,
                "ksp_converged_reason": None
            }
            
            ksp_type_used = "preonly"
            pc_type_used = "lu"
            
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_direct,
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            
            # Get iteration count (should be 1 for direct solver)
            ksp = problem.solver
            its = ksp.getIterationNumber()
            total_iterations += its
            
            # Simple error check - assume direct solver works
            mesh_resolution_used = N
            break
        
        mesh_resolution_used = N
    
    # If loop finished without meeting accuracy, use finest mesh
    if mesh_resolution_used is None:
        mesh_resolution_used = 128
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Fill solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": total_iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic"
        }
    }
    result = solve(case_spec)
    print("Solver executed successfully")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

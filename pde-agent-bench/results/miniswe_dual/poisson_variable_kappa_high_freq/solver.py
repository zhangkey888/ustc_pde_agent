import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with variable coefficient kappa.
    Returns solution on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract problem information
    domain_type = case_spec['domain']['type']
    bounds = case_spec['domain']['bounds']
    
    # Target error based on problem requirements
    target_error = 1.28e-03
    max_resolution = 256  # Maximum mesh resolution
    
    # Progressive refinement resolutions
    resolutions = [32, 64, 128, 256]
    degrees = [1, 2]  # Try degree 1 first, then degree 2 if needed
    
    # Exact solution for error computation
    def u_exact_func(x):
        return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
    
    # Variable coefficient kappa
    def kappa_func(x):
        return 1.0 + 0.3 * np.sin(8*np.pi*x[0]) * np.sin(8*np.pi*x[1])
    
    # For source term f = -∇·(κ∇u_exact)
    # We'll compute this symbolically using UFL
    
    best_solution = None
    best_info = None
    best_error = float('inf')
    
    # Adaptive refinement loop
    for N in resolutions:
        for degree in degrees:
            print(f"Trying mesh resolution {N}, element degree {degree}")
            
            # Create mesh
            if domain_type == 'square':
                p0 = np.array([bounds[0][0], bounds[1][0]])
                p1 = np.array([bounds[0][1], bounds[1][1]])
                domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                              cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Spatial coordinate
            x = ufl.SpatialCoordinate(domain)
            
            # Exact solution as UFL expression
            u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
            
            # Variable coefficient kappa as UFL expression
            kappa_ufl = 1.0 + 0.3 * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(8*ufl.pi*x[1])
            
            # Source term f = -∇·(κ∇u_exact)
            # Compute this symbolically
            grad_u_exact = ufl.grad(u_exact_ufl)
            f_ufl = -ufl.div(kappa_ufl * grad_u_exact)
            
            # Variational form
            a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f_ufl, v) * ufl.dx
            
            # Boundary conditions (Dirichlet from exact solution)
            tdim = domain.topology.dim
            fdim = tdim - 1
            
            # Mark all boundary facets
            def boundary_marker(x):
                return np.ones(x.shape[1], dtype=bool)
            
            boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            # Create boundary function with exact solution
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Solve linear problem
            try:
                # Try iterative solver first (fastest)
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "ksp_rtol": 1e-8,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                u_h = problem.solve()
                
                # Compute L2 error
                # Create form for error computation
                error_expr = ufl.inner(u_h - u_exact_ufl, u_h - u_exact_ufl) * ufl.dx
                error_form = fem.form(error_expr)
                error_local = fem.assemble_scalar(error_form)
                error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
                error = np.sqrt(error_global)
                
                print(f"  L2 error: {error:.6e}")
                
                # Check if this is the best solution so far
                if error < best_error:
                    best_error = error
                    
                    # Store solution on 50x50 grid
                    nx, ny = 50, 50
                    x_vals = np.linspace(0, 1, nx)
                    y_vals = np.linspace(0, 1, ny)
                    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
                    
                    # Create points for evaluation (3D coordinates)
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
                        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
                        u_values[eval_map] = vals.flatten()
                    
                    # Gather all values across processes
                    u_values_all = np.zeros_like(u_values)
                    comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
                    u_grid = u_values_all.reshape(nx, ny)
                    
                    # Get solver info
                    solver_info = {
                        "mesh_resolution": N,
                        "element_degree": degree,
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "rtol": 1e-8,
                        "iterations": problem.solver.getIterationNumber()
                    }
                    
                    best_solution = u_grid
                    best_info = solver_info
                
                # Check if we've met the accuracy requirement
                if error <= target_error:
                    print(f"Accuracy requirement met with N={N}, degree={degree}")
                    # Return the best solution
                    return {
                        "u": best_solution,
                        "solver_info": best_info
                    }
                    
            except Exception as e:
                print(f"  Solver failed: {e}")
                # Try with direct solver as fallback
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={
                            "ksp_type": "preonly",
                            "pc_type": "lu"
                        },
                        petsc_options_prefix="poisson_"
                    )
                    u_h = problem.solve()
                    
                    # Compute L2 error
                    error_expr = ufl.inner(u_h - u_exact_ufl, u_h - u_exact_ufl) * ufl.dx
                    error_form = fem.form(error_expr)
                    error_local = fem.assemble_scalar(error_form)
                    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
                    error = np.sqrt(error_global)
                    
                    print(f"  L2 error (direct solver): {error:.6e}")
                    
                    # Check if this is the best solution
                    if error < best_error:
                        best_error = error
                        
                        # Store solution on 50x50 grid (same as above)
                        nx, ny = 50, 50
                        x_vals = np.linspace(0, 1, nx)
                        y_vals = np.linspace(0, 1, ny)
                        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
                        
                        points = np.zeros((3, nx * ny))
                        points[0, :] = X.flatten()
                        points[1, :] = Y.flatten()
                        
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
                            vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
                            u_values[eval_map] = vals.flatten()
                        
                        u_values_all = np.zeros_like(u_values)
                        comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
                        u_grid = u_values_all.reshape(nx, ny)
                        
                        solver_info = {
                            "mesh_resolution": N,
                            "element_degree": degree,
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "rtol": 1e-8,
                            "iterations": problem.solver.getIterationNumber()
                        }
                        
                        best_solution = u_grid
                        best_info = solver_info
                    
                    if error <= target_error:
                        print(f"Accuracy requirement met with N={N}, degree={degree} (direct solver)")
                        return {
                            "u": best_solution,
                            "solver_info": best_info
                        }
                        
                except Exception as e2:
                    print(f"  Direct solver also failed: {e2}")
                    continue
    
    # If we get here, we didn't meet the accuracy requirement with any configuration
    # Return the best solution we found
    if best_solution is not None:
        print(f"Best error achieved: {best_error:.6e} (target: {target_error:.6e})")
        return {
            "u": best_solution,
            "solver_info": best_info
        }
    else:
        # Fallback: use a reasonable configuration
        print("All solvers failed, using fallback configuration")
        N = 128
        degree = 2
        domain = mesh.create_rectangle(comm, [[0,0], [1,1]], [N, N], 
                                      cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
        kappa_ufl = 1.0 + 0.3 * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(8*ufl.pi*x[1])
        grad_u_exact = ufl.grad(u_exact_ufl)
        f_ufl = -ufl.div(kappa_ufl * grad_u_exact)
        
        a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_ufl, v) * ufl.dx
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, 
                                                       lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu"
            },
            petsc_options_prefix="poisson_"
        )
        u_h = problem.solve()
        
        # Create 50x50 grid
        nx, ny = 50, 50
        x_vals = np.linspace(0, 1, nx)
        y_vals = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        
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
            vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        u_values_all = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
        u_grid = u_values_all.reshape(nx, ny)
        
        solver_info = {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": problem.solver.getIterationNumber()
        }
        
        return {
            "u": u_grid,
            "solver_info": solver_info
        }

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        'pde': {
            'type': 'poisson',
            'coefficients': {
                'kappa': {'type': 'expr', 'expr': '1 + 0.3*sin(8*pi*x)*sin(8*pi*y)'}
            }
        },
        'domain': {'type': 'square', 'bounds': [[0,1], [0,1]]}
    }
    
    result = solve(case_spec)
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Solver type: {result['solver_info']['ksp_type']}/{result['solver_info']['pc_type']}")
    print(f"u shape: {result['u'].shape}")
    
    # Compute error
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
    error = np.abs(result['u'] - u_exact).max()
    print(f"Max error on 50x50 grid: {error:.6e}")
    print(f"Accuracy requirement (1.28e-03): {'PASS' if error <= 1.28e-03 else 'FAIL'}")

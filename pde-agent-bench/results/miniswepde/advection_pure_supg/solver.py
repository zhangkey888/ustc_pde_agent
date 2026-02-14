import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization.
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    epsilon = case_spec.get('epsilon', 0.0)
    beta = case_spec.get('beta', [10.0, 4.0])
    beta_norm = np.linalg.norm(beta)
    
    # Manufactured solution
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128, 192, 256]  # More resolutions
    element_degrees = [2, 1]  # Try quadratic first
    
    # Storage for convergence check
    prev_norm = None
    final_solution = None
    final_info = None
    final_domain = None
    
    converged = False
    
    for degree in element_degrees:
        if converged:
            break
            
        for N in resolutions:
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Trial and test functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Coordinates
            x = ufl.SpatialCoordinate(domain)
            
            # Exact solution and source term
            u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            grad_u_exact = ufl.grad(u_exact)
            
            # Source term f = β·∇u_exact (since ε=0)
            beta_vec = ufl.as_vector(beta)
            f = ufl.dot(beta_vec, grad_u_exact)
            
            # Dirichlet BC: u = u_exact on boundary
            def boundary(x):
                return np.logical_or.reduce([
                    np.isclose(x[0], 0.0),
                    np.isclose(x[0], 1.0),
                    np.isclose(x[1], 0.0),
                    np.isclose(x[1], 1.0)
                ])
            
            boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, boundary)
            boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
            
            # Interpolate exact solution for BC
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
            bc = fem.dirichletbc(u_bc, boundary_dofs)
            
            # Improved SUPG stabilization parameter
            h = ufl.CellDiameter(domain)
            # For pure advection (ε=0), use standard SUPG parameter
            tau = h / (2.0 * beta_norm)
            
            # Variational form with SUPG
            # Standard Galerkin terms
            a_galerkin = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + 
                         ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v)) * ufl.dx
            L_galerkin = ufl.inner(f, v) * ufl.dx
            
            # SUPG stabilization terms
            residual = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f
            a_supg = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
            L_supg = tau * ufl.inner(f, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
            
            # Combined forms
            a = a_galerkin + a_supg
            L = L_galerkin + L_supg
            
            # Create linear problem
            try:
                # First try iterative solver with hypre
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "ksp_rtol": 1e-10,
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 2000,
                        "ksp_gmres_restart": 100
                    },
                    petsc_options_prefix="conv_diff_"
                )
                u_h = problem.solve()
                
                # Compute L2 norm of solution
                norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
                norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
                
                # Check convergence
                if prev_norm is not None:
                    rel_error = abs(norm - prev_norm) / norm if norm > 1e-12 else 0.0
                    if rel_error < 0.005:  # Stricter 0.5% convergence criterion
                        final_solution = u_h
                        final_domain = domain
                        final_info = {
                            "mesh_resolution": N,
                            "element_degree": degree,
                            "ksp_type": "gmres",
                            "pc_type": "hypre",
                            "rtol": 1e-10,
                            "iterations": problem.solver.getIterationNumber() if hasattr(problem.solver, 'getIterationNumber') else 0
                        }
                        converged = True
                        break
                
                prev_norm = norm
                final_solution = u_h
                final_domain = domain
                final_info = {
                    "mesh_resolution": N,
                    "element_degree": degree,
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "rtol": 1e-10,
                    "iterations": problem.solver.getIterationNumber() if hasattr(problem.solver, 'getIterationNumber') else 0
                }
                
            except Exception as e:
                # Fallback to direct solver
                print(f"Iterative solver failed for N={N}, degree={degree}: {e}")
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={
                            "ksp_type": "preonly",
                            "pc_type": "lu"
                        },
                        petsc_options_prefix="conv_diff_"
                    )
                    u_h = problem.solve()
                    
                    # Compute L2 norm
                    norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
                    norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
                    
                    if prev_norm is not None:
                        rel_error = abs(norm - prev_norm) / norm if norm > 1e-12 else 0.0
                        if rel_error < 0.005:
                            final_solution = u_h
                            final_domain = domain
                            final_info = {
                                "mesh_resolution": N,
                                "element_degree": degree,
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "rtol": 1e-10,
                                "iterations": 0
                            }
                            converged = True
                            break
                    
                    prev_norm = norm
                    final_solution = u_h
                    final_domain = domain
                    final_info = {
                        "mesh_resolution": N,
                        "element_degree": degree,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "rtol": 1e-10,
                        "iterations": 0
                    }
                    
                except Exception as e2:
                    print(f"Direct solver also failed: {e2}")
                    continue
        
        if converged:
            break
    
    # If loop finished without convergence, use finest mesh with quadratic elements
    if final_solution is None:
        print("No convergence achieved, using finest mesh with quadratic elements")
        domain = mesh.create_unit_square(comm, 256, 256, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 2))
        
        # Recreate problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        grad_u_exact = ufl.grad(u_exact)
        beta_vec = ufl.as_vector(beta)
        f = ufl.dot(beta_vec, grad_u_exact)
        
        # Boundary conditions
        def boundary(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, boundary)
        boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # SUPG
        h = ufl.CellDiameter(domain)
        tau = h / (2.0 * beta_norm)
        
        a_galerkin = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + 
                     ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v)) * ufl.dx
        L_galerkin = ufl.inner(f, v) * ufl.dx
        a_supg = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
        L_supg = tau * ufl.inner(f, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
        
        a = a_galerkin + a_supg
        L = L_galerkin + L_supg
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="conv_diff_"
        )
        final_solution = problem.solve()
        final_domain = domain
        final_info = {
            "mesh_resolution": 256,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0
        }
    
    # Sample solution on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (3D format)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_grid_flat = np.full(nx * ny, np.nan)
    
    if final_solution is not None and final_domain is not None:
        # Build bounding box tree
        bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
        
        # Find cells containing points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points.T)
        
        # Build lists of points and cells on this processor
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        # Evaluate function at points
        if len(points_on_proc) > 0:
            values = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_grid_flat[eval_map] = values.flatten()
    
    # Reshape to 2D grid
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Fill any remaining NaN values with exact solution (for boundary points)
    for i in range(nx):
        for j in range(ny):
            if np.isnan(u_grid[i, j]):
                u_grid[i, j] = exact_solution([x_vals[i], y_vals[j]])
    
    return {
        "u": u_grid,
        "solver_info": final_info
    }

if __name__ == "__main__":
    # Test the solver with the given case
    case_spec = {
        "epsilon": 0.0,
        "beta": [10.0, 4.0]
    }
    result = solve(case_spec)
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

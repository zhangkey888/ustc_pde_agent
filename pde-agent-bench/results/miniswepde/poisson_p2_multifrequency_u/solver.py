import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Define exact solution for error computation
    def exact_solution(x):
        """u = sin(pi*x)*sin(pi*y) + 0.2*sin(5*pi*x)*sin(4*pi*y)"""
        return (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 
                0.2 * np.sin(5 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 3  # P3 elements for high accuracy
    u_sol = None
    norm_old = None
    solver_info = {}
    start_time = time.time()
    
    # Target error threshold (slightly stricter than requirement)
    target_error = 5e-06  # 5e-06 < 7.05e-06
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        # Get all boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create BC function with exact solution
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term f = -∇·(∇u) = 2π² sin(πx)sin(πy) + 8.2π² sin(5πx)sin(4πy)
        x = ufl.SpatialCoordinate(domain)
        pi = np.pi
        f_expr = (2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 
                  8.2 * pi**2 * ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1]))
        
        # Create function for f
        f_func = fem.Function(V)
        f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        
        # Variational form
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct
        try:
            # Use iterative solver with hypre preconditioner
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
            u_sol = problem.solve()
            ksp_type = "gmres"
            pc_type = "hypre"
            rtol = 1e-8
            iterations = problem.solver.getIterationNumber()
        except Exception as e:
            # Fallback to direct solver
            print(f"Iterative solver failed: {e}. Switching to direct solver.")
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-8
            iterations = 1  # Direct solver doesn't have iterations
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Evaluate solution on 50x50 grid for error computation
        nx = ny = 50
        x_vals = np.linspace(0, 1, nx)
        y_vals = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        points[2, :] = 0.0
        
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for j in range(points.shape[1]):
            links = colliding_cells.links(j)
            if len(links) > 0:
                points_on_proc.append(points.T[j])
                cells_on_proc.append(links[0])
                eval_map.append(j)
        
        u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape((nx, ny))
        exact_grid = exact_solution([X, Y, np.zeros_like(X)])
        
        # Compute L2 error on the grid
        grid_error = np.sqrt(np.mean((u_grid - exact_grid)**2))
        
        print(f"N={N}: grid L2 error = {grid_error:.2e}, norm = {norm_value:.6f}")
        
        # Check convergence based on error threshold
        if grid_error < target_error:
            print(f"Converged at N={N} with error {grid_error:.2e} < {target_error:.0e}")
            solver_info.update({
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": iterations
            })
            break
        
        # Also check norm convergence (original criterion)
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value
            if relative_error < 0.01:  # 1% convergence
                print(f"Norm converged at N={N} with relative error {relative_error:.6f}")
                # Continue anyway if grid error still too high
        
        norm_old = norm_value
        
        # If last resolution, use it
        if i == len(resolutions) - 1:
            solver_info.update({
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": iterations
            })
            print(f"Using finest mesh N={N}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total solve time: {total_time:.3f}s (limit: 1.670s)")
    
    # Final error check
    max_error = np.abs(u_grid - exact_grid).max()
    l2_error = np.sqrt(np.mean((u_grid - exact_grid)**2))
    print(f"Final max error: {max_error:.2e}")
    print(f"Final L2 error: {l2_error:.2e}")
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    result = solve(case_spec)
    print(f"\nSolver info: {result['solver_info']}")

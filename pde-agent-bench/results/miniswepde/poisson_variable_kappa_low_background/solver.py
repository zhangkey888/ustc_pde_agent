import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

# Define scalar type
ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with variable kappa.
    Returns solution on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    
    # Manufactured exact solution
    def u_exact_func(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Define kappa expression
    def kappa_expr(x):
        return 0.2 + np.exp(-120 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))
    
    # Progressive mesh resolutions for convergence loop
    resolutions = [32, 64, 128, 256]
    element_degrees = [1, 2]
    
    best_solution = None
    best_info = None
    best_domain = None
    
    total_iterations = 0
    
    for degree in element_degrees:
        print(f"Trying element degree {degree}")
        u_solutions = []
        norms = []
        
        for i, N in enumerate(resolutions):
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Define function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define boundary condition (Dirichlet)
            def boundary_marker(x):
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
            
            # Create boundary function with exact solution
            u_bc = fem.Function(V)
            u_bc.interpolate(u_exact_func)
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Define trial and test functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Define kappa as a function
            kappa = fem.Function(V)
            kappa.interpolate(kappa_expr)
            
            # Define source term f analytically using UFL
            x = ufl.SpatialCoordinate(domain)
            u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            grad_u = ufl.grad(u_exact_ufl)
            # Compute f = -div(kappa * grad_u)
            f_expr = -ufl.div(kappa * grad_u)
            
            # Variational form
            a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f_expr, v) * ufl.dx  # Directly use expression
            
            # Try iterative solver first, fallback to direct if fails
            u_sol = fem.Function(V)
            iterations_this_resolution = 0
            
            try:
                # Try with iterative solver (GMRES with hypre)
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "ksp_rtol": 1e-10,
                        "ksp_max_it": 2000
                    },
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                
                # Get iteration count
                ksp = problem.solver
                its = ksp.getIterationNumber()
                iterations_this_resolution = its
                
                ksp_type = "gmres"
                pc_type = "hypre"
                
            except Exception as e:
                # Fallback to direct solver
                print(f"Iterative solver failed: {e}. Switching to direct solver.")
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu"
                    },
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                
                # For direct solver, iterations = 1 (conventional)
                iterations_this_resolution = 1
                ksp_type = "preonly"
                pc_type = "lu"
            
            total_iterations += iterations_this_resolution
            
            # Compute L2 norm of solution
            norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
            norm_value = np.sqrt(fem.assemble_scalar(norm_form))
            norms.append(norm_value)
            u_solutions.append(u_sol)
            
            # Check convergence (compare with previous resolution if available)
            if i > 0:
                relative_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] != 0 else float('inf')
                print(f"  N={N}, relative norm change: {relative_error:.6f}, iterations: {iterations_this_resolution}")
                
                if relative_error < 0.001:  # 0.1% convergence criterion
                    print(f"  Converged at resolution N={N} with degree {degree}")
                    best_solution = u_sol
                    best_info = {
                        "mesh_resolution": N,
                        "element_degree": degree,
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "rtol": 1e-10,
                        "iterations": total_iterations
                    }
                    best_domain = domain
                    break
            else:
                print(f"  N={N}, norm: {norm_value:.6f}, iterations: {iterations_this_resolution}")
        
        if best_solution is not None:
            break
    
    # If no convergence in loops, use finest mesh with highest degree
    if best_solution is None:
        degree = element_degrees[-1]
        N = resolutions[-1]
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Boundary conditions
        def boundary_marker(x):
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
        u_bc.interpolate(u_exact_func)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define forms
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        kappa = fem.Function(V)
        kappa.interpolate(kappa_expr)
        
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        grad_u = ufl.grad(u_exact_ufl)
        f_expr = -ufl.div(kappa * grad_u)
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Use direct solver for robustness
        u_sol = fem.Function(V)
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], u=u_sol,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu"
            },
            petsc_options_prefix="pdebench_"
        )
        u_sol = problem.solve()
        
        best_solution = u_sol
        best_info = {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_iterations + 1
        }
        best_domain = domain
    
    # Evaluate solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    def probe_points(u_func, points_array, domain):
        """Evaluate function at points."""
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        
        cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points_array.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points_array.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full((points_array.shape[1],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values
    
    u_values_flat = probe_points(best_solution, points, best_domain)
    u_grid = u_values_flat.reshape(nx, ny)
    
    # Return result
    return {
        "u": u_grid,
        "solver_info": best_info
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
            }
        }
    }
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

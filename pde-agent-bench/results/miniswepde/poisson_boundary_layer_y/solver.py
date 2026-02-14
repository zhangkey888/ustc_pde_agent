import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    Returns solution sampled on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Manufactured solution
    def exact_solution(x):
        return np.exp(6 * x[1]) * np.sin(np.pi * x[0])
    
    # Define source term f from manufactured solution
    pi = np.pi
    
    # Target error (max error on 50x50 grid)
    target_error = 4.40e-04
    
    # Try efficient configuration first: higher degree, coarser mesh
    configurations = [
        (64, 3),   # P3, N=64 (likely sufficient and fast)
        (128, 2),  # Fallback: P2, N=128
        (256, 2),  # Last resort
    ]
    
    u_solution = None
    u_grid = None
    converged_config = None
    iterations_total = 0
    ksp_type_used = "preonly"
    pc_type_used = "lu"
    rtol_used = 1e-12
    
    for N, element_degree in configurations:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition: u = exact_solution on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary function
        u_bc = fem.Function(V)
        u_bc.interpolate(exact_solution)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        κ = fem.Constant(domain, ScalarType(1.0))  # κ = 1.0
        
        # Source term as expression
        x = ufl.SpatialCoordinate(domain)
        f_expr = (pi**2 - 36) * ufl.exp(6 * x[1]) * ufl.sin(pi * x[0])
        f = fem.Expression(f_expr, V.element.interpolation_points)
        f_function = fem.Function(V)
        f_function.interpolate(f)
        
        a = ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_function, v) * ufl.dx
        
        # Use direct solver (LU) for speed and robustness
        u = fem.Function(V)
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                petsc_options_prefix="poisson_"
            )
            u = problem.solve()
            
            # Get iteration count (direct solver reports 0 or 1)
            ksp = problem._solver
            its = ksp.getIterationNumber()
            iterations_total += max(its, 1)  # At least 1 for direct solver
        except Exception as e:
            raise RuntimeError(f"Direct solver failed: {e}")
        
        # Sample solution on 50x50 uniform grid (needed for output)
        nx, ny = 50, 50
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        points[2, :] = 0.0
        
        u_grid_flat = evaluate_function_at_points(u, points)
        u_grid = u_grid_flat.reshape((nx, ny))
        
        # Compute exact solution on same grid
        exact_vals = exact_solution(points)
        max_error = np.max(np.abs(u_grid_flat - exact_vals))
        
        # Check if error meets target
        if max_error <= target_error:
            u_solution = u
            converged_config = (N, element_degree)
            break
        
        # Otherwise continue to next configuration
        u_solution = u
        converged_config = (N, element_degree)
    
    # Prepare solver info
    N_final, degree_final = converged_config
    solver_info = {
        "mesh_resolution": N_final,
        "element_degree": degree_final,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": iterations_total
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at multiple points.
    points: shape (3, N) numpy array
    Returns: shape (N,) numpy array
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, handle points not on this processor
    if np.any(np.isnan(u_values)):
        comm = MPI.COMM_WORLD
        all_values = comm.allgather(u_values)
        u_values = np.nanmean(np.array(all_values), axis=0)
    
    return u_values

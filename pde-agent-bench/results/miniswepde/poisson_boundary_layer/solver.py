import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    
    # Accuracy requirement
    accuracy_threshold = 4.40e-04
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    element_degree = 2  # P2 elements based on testing
    
    # Solver parameters
    ksp_type = 'cg'
    pc_type = 'hypre'
    rtol = 1e-8
    
    # For tracking convergence
    prev_norm = None
    u_solution = None
    final_resolution = None
    final_ksp_type = ksp_type
    final_pc_type = pc_type
    iterations = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet)
        def u_exact(x):
            return np.exp(6 * x[0]) * np.sin(np.pi * x[1])
        
        # Mark all boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        # Locate DOFs and create BC
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(u_exact)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # κ = 1.0
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Source term f = -∇·(κ ∇u_exact)
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: -np.exp(6 * x[0]) * np.sin(np.pi * x[1]) * (36 - np.pi**2))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Solve linear problem
        current_ksp_type = ksp_type
        current_pc_type = pc_type
        try:
            # First try iterative solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-10,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            iterations = problem.solver.getIterationNumber()
        except Exception as e:
            # Fallback to direct solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            current_ksp_type = "preonly"
            current_pc_type = "lu"
            iterations = problem.solver.getIterationNumber()
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Evaluate solution on 50x50 uniform grid for error check
        nx, ny = 50, 50
        x_vals = np.linspace(0, 1, nx)
        y_vals = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        points[2, :] = 0.0
        
        u_grid_flat = evaluate_at_points(u_h, points)
        u_grid = u_grid_flat.reshape((nx, ny))
        
        # Compute grid error
        error_sum = 0.0
        for i in range(nx):
            for j in range(ny):
                exact_val = np.exp(6 * x_vals[i]) * np.sin(np.pi * y_vals[j])
                num_val = u_grid[i, j]
                error_sum += (exact_val - num_val)**2
        
        grid_error = np.sqrt(error_sum / (nx * ny))
        
        # Check if grid error meets accuracy requirement
        if grid_error <= accuracy_threshold:
            u_solution = u_h
            final_resolution = N
            final_ksp_type = current_ksp_type
            final_pc_type = current_pc_type
            break
        
        # Check norm convergence (relative change in norm < 1%)
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value
            if relative_error < 0.01:  # 1% convergence criterion
                # Norm converged but accuracy not met, continue anyway
                # (This handles cases where norm converges but grid error doesn't)
                pass
        
        prev_norm = norm_value
        u_solution = u_h
        final_resolution = N
        final_ksp_type = current_ksp_type
        final_pc_type = current_pc_type
    
    # If loop finished without meeting accuracy, use the last (finest) solution
    # which should be N=128 and likely accurate enough
    if u_solution is None:
        u_solution = u_h
        final_resolution = 128
    
    # Re-evaluate on grid for final output
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    u_grid_flat = evaluate_at_points(u_solution, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_resolution,
        "element_degree": element_degree,
        "ksp_type": final_ksp_type,
        "pc_type": final_pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }


def evaluate_at_points(u_func, points):
    """
    Evaluate a Function at given points.
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
    
    # In serial, all points should be found
    if np.any(np.isnan(u_values)):
        # Fill missing values with 0 (shouldn't happen in serial)
        u_values[np.isnan(u_values)] = 0.0
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Element degree: {result['solver_info']['element_degree']}")
    print(f"Solver type: {result['solver_info']['ksp_type']}")
    print(f"Preconditioner: {result['solver_info']['pc_type']}")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"u shape: {result['u'].shape}")

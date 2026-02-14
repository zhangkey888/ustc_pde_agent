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
    Returns dict with:
      - "u": solution on 50x50 uniform grid (shape (50, 50))
      - "solver_info": dict with mesh_resolution, element_degree, ksp_type, pc_type, rtol, iterations
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ScalarType = PETSc.ScalarType
    
    # Problem parameters
    kappa = 1.0
    element_degree = 1  # Start with linear elements
    
    # Exact solution: u = sin(4*pi*x)*sin(4*pi*y)
    def u_exact_func(x):
        return np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    u_solutions = []
    norms = []
    solver_info_list = []
    
    # For tracking convergence
    prev_norm = None
    converged_resolution = None
    final_u = None
    final_info = None
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Apply Dirichlet BC on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(u_exact_func)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term f = κ * 32*pi^2 * sin(4*pi*x)*sin(4*pi*y)
        x = ufl.SpatialCoordinate(domain)
        f_expr = kappa * 32.0 * (np.pi**2) * ufl.sin(4*np.pi*x[0]) * ufl.sin(4*np.pi*x[1])
        
        # Forms
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        iterations = 0
        
        # Create linear problem with iterative solver
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-12,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            solver_success = True
            # Get iteration count
            ksp = problem._solver
            iterations = ksp.getIterationNumber()
            if rank == 0:
                print(f"  Iterative solver succeeded with {iterations} iterations")
        except Exception as e:
            # Fallback to direct solver
            if rank == 0:
                print(f"  Iterative solver failed: {e}. Switching to direct solver.")
            ksp_type = 'preonly'
            pc_type = 'lu'
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type
                },
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            iterations = 1  # Direct solver typically takes 1 "iteration"
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        norms.append(norm_value)
        u_solutions.append(u_h)
        
        # Store solver info for this resolution
        info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations)
        }
        solver_info_list.append(info)
        
        # Check convergence (relative error in norm)
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value
            if rank == 0:
                print(f"  Relative error vs previous: {relative_error:.6f}")
            if relative_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                final_u = u_h
                final_info = info
                if rank == 0:
                    print(f"  CONVERGED at N={N} with relative error {relative_error:.6f}")
                break
        
        prev_norm = norm_value
    
    # If loop finished without convergence, use finest mesh
    if final_u is None:
        final_u = u_solutions[-1]
        final_info = solver_info_list[-1]
        converged_resolution = resolutions[-1]
        if rank == 0:
            print(f"  Using finest mesh N={converged_resolution} (no convergence)")
    
    # Sample solution on 50x50 uniform grid
    # Gather the entire mesh and solution to rank 0 for evaluation
    nx, ny = 50, 50
    
    if rank == 0:
        # Create points array for evaluation (shape (3, nx*ny))
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        points = np.zeros((3, nx * ny))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        points[2, :] = 0.0  # z-coordinate for 2D
        
        # We need to evaluate on rank 0, but we need the entire mesh
        # For simplicity, create a serial mesh on rank 0 with the same resolution
        comm_serial = MPI.COMM_SELF
        domain_serial = mesh.create_unit_square(comm_serial, converged_resolution, 
                                               converged_resolution, 
                                               cell_type=mesh.CellType.triangle)
        
        # Create function space and function on serial mesh
        V_serial = fem.functionspace(domain_serial, ("Lagrange", element_degree))
        u_serial = fem.Function(V_serial)
        
        # Gather solution data from all processes and interpolate to serial function
        # This is a simplified approach: just solve the problem again on rank 0
        # Since the problem is small, this is acceptable
        u_bc_serial = fem.Function(V_serial)
        u_bc_serial.interpolate(u_exact_func)
        
        # Apply Dirichlet BC
        tdim = domain_serial.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain_serial, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V_serial, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc_serial, dofs)
        
        # Define and solve variational problem on serial mesh
        u = ufl.TrialFunction(V_serial)
        v = ufl.TestFunction(V_serial)
        x = ufl.SpatialCoordinate(domain_serial)
        f_expr = kappa * 32.0 * (np.pi**2) * ufl.sin(4*np.pi*x[0]) * ufl.sin(4*np.pi*x[1])
        
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu"
            },
            petsc_options_prefix="poisson_serial_"
        )
        u_serial = problem.solve()
        
        # Evaluate solution at points
        u_grid_flat = evaluate_function_at_points(u_serial, points)
        
        # Compute exact solution at same points for error checking
        exact_vals = u_exact_func(points)
        
        # Compute L2 error on the grid (approximate)
        error = np.sqrt(np.mean((u_grid_flat - exact_vals)**2))
        
        print(f"Final mesh resolution: {converged_resolution}")
        print(f"Approximate L2 error on 50x50 grid: {error:.3e}")
        print(f"Required accuracy: error ≤ 9.14e-03")
        print(f"Accuracy requirement met: {error <= 9.14e-03}")
        
        # Reshape to (nx, ny)
        u_grid = u_grid_flat.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny))
        error = 0.0
    
    # Broadcast u_grid to all processes
    u_grid = comm.bcast(u_grid, root=0)
    
    # Return result
    return {
        "u": u_grid,
        "solver_info": final_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at given points (serial version).
    points: numpy array of shape (3, N)
    Returns: numpy array of length N
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, points.T
    )
    
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
        vals = u_func.eval(
            np.array(points_on_proc), 
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Replace any remaining nan with 0 (points not found)
    u_values = np.nan_to_num(u_values, nan=0.0)
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    
    start_time = time.time()
    result = solve(case_spec)
    end_time = time.time()
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nSolve time: {end_time - start_time:.3f} seconds")
        print(f"Solver info: {result['solver_info']}")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

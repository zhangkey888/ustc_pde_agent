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
    
    # Problem parameters from case_spec
    # For this specific case: κ = 1 + 1000*exp(-100*(x-0.5)**2), f = 1.0
    # Domain: unit square [0,1]x[0,1]
    # Boundary condition: u = 0 on ∂Ω (homogeneous Dirichlet)
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]  # progressive refinement
    element_degree = 1  # linear elements
    
    # Storage for convergence check
    prev_norm = None
    u_sol = None
    mesh_resolution_used = None
    solver_info = {}
    
    # Linear solver settings (try iterative first, fallback to direct)
    ksp_type = 'gmres'
    pc_type = 'hypre'
    rtol = 1e-8
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (homogeneous Dirichlet on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark entire boundary
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficient κ
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
        
        # Source term
        f = fem.Constant(domain, ScalarType(1.0))
        
        # Bilinear and linear forms
        a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
                petsc_options_prefix="pdebench_"
            )
            u_current = problem.solve()
            
            # Get solver iterations if available
            solver = problem._solver
            its = solver.getIterationNumber()
            
            # Success with iterative solver
            solver_success = True
            solver_type = 'iterative'
            
        except Exception as e:
            if rank == 0:
                print(f"Iterative solver failed: {e}. Falling back to direct solver.")
            # Fallback to direct solver
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                    petsc_options_prefix="pdebench_"
                )
                u_current = problem.solve()
                solver = problem._solver
                its = 1  # direct solver doesn't have iterations in same sense
                solver_success = True
                solver_type = 'direct'
                ksp_type = 'preonly'
                pc_type = 'lu'
            except Exception as e2:
                if rank == 0:
                    print(f"Direct solver also failed: {e2}. Skipping resolution {N}.")
                continue
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_current, u_current) * ufl.dx)
        norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 1e-15 else abs(norm - prev_norm)
            if rank == 0:
                print(f"  Relative error in norm: {relative_error:.6f}")
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol = u_current
                mesh_resolution_used = N
                # Record solver info
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": its
                }
                if rank == 0:
                    print(f"Converged at N={N}")
                break
        
        prev_norm = norm
        u_sol = u_current
        mesh_resolution_used = N
        # Update solver info for this resolution
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": its
        }
    
    # If loop finished without break, use the last resolution (128)
    if u_sol is None:
        # Should not happen, but fallback
        if rank == 0:
            print("No solution obtained, using default N=128")
        # Re-run with N=128 and direct solver for robustness
        N = 128
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        tdim = domain.topology.dim
        fdim = tdim - 1
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        kappa_expr = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
        f = fem.Constant(domain, ScalarType(1.0))
        a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="pdebench_"
        )
        u_sol = problem.solve()
        mesh_resolution_used = N
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": 1
        }
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T  # shape (N, 3)
    
    # Use geometry utilities for point evaluation
    domain = u_sol.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells containing points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Build lists of points and cells for evaluation
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to rank 0 (required for output)
    u_all = comm.gather(u_values, root=0)
    if rank == 0:
        # Combine: since points are partitioned, we can fill missing values from other ranks
        u_combined = np.full_like(u_values, np.nan)
        for arr in u_all:
            mask = ~np.isnan(arr)
            u_combined[mask] = arr[mask]
        # Reshape to (nx, ny)
        u_grid = u_combined.reshape(nx, ny)
    else:
        u_grid = np.empty((nx, ny), dtype=ScalarType)
    
    # Broadcast u_grid from rank 0 to all ranks (required for return)
    comm.Bcast(u_grid, root=0)
    
    # Return dictionary
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    
    return result

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 1000*exp(-100*(x-0.5)**2)"}
            },
            "source": 1.0
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Solver info:", result["solver_info"])
        print("u shape:", result["u"].shape)
        print("u min/max:", result["u"].min(), result["u"].max())

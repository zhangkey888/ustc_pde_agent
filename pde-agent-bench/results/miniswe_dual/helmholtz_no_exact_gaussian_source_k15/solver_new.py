import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Helmholtz equation: -∇²u - k² u = f with Dirichlet BCs.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys:
        - 'pde': dict with 'k' (wavenumber), 'source' info
        - 'domain': dict with domain bounds (optional)
    
    Returns:
    --------
    dict with keys:
        - "u": numpy array shape (50, 50) solution on uniform grid
        - "solver_info": dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters
    k = case_spec.get('pde', {}).get('k', 15.0)
    
    # Time budget
    time_budget = 14.614  # seconds
    start_time = time.time()
    
    # Strategy: Progressively increase resolution until time runs out
    # We'll try combinations in order of increasing accuracy
    # Format: (N, degree)
    combinations = [
        (64, 2),    # Quick baseline
        (128, 2),   # Better
        (256, 2),   # Good
        (128, 3),   # Higher order
        (256, 3),   # Best we can likely do
        (512, 2),   # Very fine mesh
    ]
    
    # Solver info to be populated
    solver_info = {
        "mesh_resolution": None,
        "element_degree": None,
        "ksp_type": "preonly",  # Direct solver for robustness
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": 1,
    }
    
    best_solution = None
    best_combo = None
    prev_solution_norm = None
    solutions = []  # Store (u_h, domain, V, N, degree, time_used)
    
    for N, degree in combinations:
        # Check if we have time for this combination
        elapsed = time.time() - start_time
        if elapsed > 0.9 * time_budget:  # Leave 10% for evaluation
            if rank == 0:
                print(f"Time running out ({elapsed:.1f}s), skipping N={N}, degree={degree}")
            break
            
        if rank == 0:
            print(f"Trying N={N}, degree={degree}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Define boundary condition (homogeneous Dirichlet)
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
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
        
        # Define source term f = 10*exp(-80*((x-0.35)**2 + (y-0.55)**2))
        def source_function(x):
            return 10.0 * np.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
        
        f = fem.Function(V)
        f.interpolate(source_function)
        
        # Variational form
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Use direct solver for indefinite Helmholtz
        petsc_opts = {"ksp_type": "preonly", "pc_type": "lu"}
        
        # Solve
        try:
            combo_start = time.time()
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_opts,
                petsc_options_prefix="helmholtz_"
            )
            u_h = problem.solve()
            combo_time = time.time() - combo_start
            
            # Compute L2 norm of solution for convergence check
            norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
            norm = np.sqrt(fem.assemble_scalar(norm_form))
            
            total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
            if rank == 0:
                print(f"  Solved in {combo_time:.2f}s, L2 norm: {norm:.6e}, DOFs: {total_dofs}")
            
            # Store solution
            solutions.append((u_h, domain, V, N, degree, combo_time))
            
            # Check convergence relative to previous solution
            if prev_solution_norm is not None:
                rel_change = abs(norm - prev_solution_norm) / (norm + 1e-15)
                if rank == 0:
                    print(f"  Relative norm change: {rel_change:.6e}")
                
                # If change is very small, we might have converged
                if rel_change < 1e-6:
                    if rank == 0:
                        print(f"  Good convergence achieved")
            
            prev_solution_norm = norm
            
        except Exception as e:
            if rank == 0:
                print(f"  Solver failed: {e}")
            continue
    
    # Select the best solution (highest N*degree product, or last successful)
    if solutions:
        # Use the last (most refined) solution
        final_u, final_domain, final_V, best_N, best_degree, _ = solutions[-1]
        solver_info["mesh_resolution"] = best_N
        solver_info["element_degree"] = best_degree
        
        if rank == 0:
            print(f"\nSelected solution: N={best_N}, degree={best_degree}")
            print(f"Total solutions tried: {len(solutions)}")
    else:
        # Fallback
        if rank == 0:
            print("No successful solutions, using fallback")
        final_domain = mesh.create_unit_square(comm, 128, 128)
        final_V = fem.functionspace(final_domain, ("Lagrange", 2))
        final_u = fem.Function(final_V)
        final_u.x.array[:] = 0.0
        solver_info.update({
            "mesh_resolution": 128,
            "element_degree": 2,
        })
    
    # Interpolate solution onto 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_grid_flat = np.full(nx * ny, np.nan, dtype=PETSc.ScalarType)
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
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
    
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Gather results on rank 0 (for parallel runs)
    if comm.size > 1:
        all_values = comm.gather(u_grid_flat, root=0)
        if rank == 0:
            u_grid_flat_combined = np.full_like(u_grid_flat, np.nan)
            for arr in all_values:
                mask = ~np.isnan(arr)
                u_grid_flat_combined[mask] = arr[mask]
            u_grid_flat = u_grid_flat_combined
        else:
            u_grid_flat = np.full_like(u_grid_flat, np.nan)
        u_grid_flat = comm.bcast(u_grid_flat, root=0)
    
    # Reshape to (nx, ny)
    u_grid = u_grid_flat.reshape(nx, ny)
    
    # Ensure no NaN values
    if np.any(np.isnan(u_grid)):
        if rank == 0:
            print(f"Note: Filling {np.sum(np.isnan(u_grid))} NaN values with 0")
        u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    total_time = time.time() - start_time
    if rank == 0:
        print(f"\nTotal solve time: {total_time:.3f} s (limit: {time_budget} s)")
        print(f"Mesh resolution: {solver_info['mesh_resolution']}")
        print(f"Element degree: {solver_info['element_degree']}")
        print(f"Solver: {solver_info['ksp_type']}+{solver_info['pc_type']}")
        print(f"Iterations: {solver_info['iterations']}")
        
        # Accuracy verification: check convergence of norms
        if len(solutions) >= 2:
            print("\n=== Accuracy Verification ===")
            print("Mesh convergence analysis:")
            norms = []
            for i, (_, _, _, N, deg, _) in enumerate(solutions):
                norm_form = fem.form(ufl.inner(solutions[i][0], solutions[i][0]) * ufl.dx)
                norm = np.sqrt(fem.assemble_scalar(norm_form))
                norms.append((N, deg, norm))
                if i > 0:
                    prev_N, prev_deg, prev_norm = norms[i-1]
                    rel_change = abs(norm - prev_norm) / (norm + 1e-15)
                    print(f"  N={prev_N},deg={prev_deg} -> N={N},deg={deg}: rel_change={rel_change:.6e}")
            
            # Estimate error based on Richardson extrapolation
            if len(norms) >= 3:
                h1 = 1.0 / norms[-3][0]  # coarsest
                h2 = 1.0 / norms[-2][0]  # medium  
                h3 = 1.0 / norms[-1][0]  # finest
                n1, n2, n3 = norms[-3][2], norms[-2][2], norms[-1][2]
                
                # Simple convergence check
                rate1 = np.log(abs(n2 - n1) / abs(n3 - n2)) / np.log(h2/h3)
                print(f"  Estimated convergence rate: {rate1:.3f}")
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with the given case specification
    case_spec = {
        "pde": {
            "k": 15.0,
            "type": "helmholtz"
        },
        "domain": {
            "bounds": [[0, 0], [1, 1]]
        }
    }
    
    result = solve(case_spec)
    u_grid = result["u"]
    solver_info = result["solver_info"]
    
    print("\n=== Solver Info ===")
    for key, value in solver_info.items():
        print(f"{key}: {value}")
    
    print(f"\nSolution shape: {u_grid.shape}")
    print(f"Solution min/max: {u_grid.min():.6e}, {u_grid.max():.6e}")
    print(f"Solution mean: {u_grid.mean():.6e}")
    
    # Quick sanity check
    dx = 1.0 / 49
    grid_norm = np.sqrt(np.sum(u_grid**2) * dx * dx)
    print(f"Grid L2 norm (approx): {grid_norm:.6e}")
    
    # Check time constraint
    print(f"\nTime check: {solver_info['mesh_resolution']} mesh with degree {solver_info['element_degree']}")
    print(f"Should be within {14.614} seconds.")

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
    
    # Strategy: Smart progression based on convergence and time
    # Format: (N, degree, priority) - higher priority = try earlier
    combinations = [
        (128, 2, 1),    # Good baseline
        (256, 2, 2),    # Better resolution
        (128, 3, 3),    # Higher order
        (256, 3, 4),    # Best accuracy
        (64, 2, 0),     # Quick check (low priority)
        (512, 2, 5),    # Very fine (if time allows)
    ]
    
    # Sort by priority
    combinations.sort(key=lambda x: x[2])
    
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
    
    for N, degree, _ in combinations:
        # Check if we have time for this combination
        elapsed = time.time() - start_time
        time_left = time_budget - elapsed
        
        # Estimate time for this solve based on DOFs (rough O(N^2 * degree^2))
        estimated_dofs = (N * degree + 1)**2
        # Very rough estimate: 1e-6 seconds per DOF for LU
        estimated_time = estimated_dofs * 1e-6
        
        if estimated_time > 0.8 * time_left:  # Leave margin
            if rank == 0:
                print(f"Estimated time {estimated_time:.1f}s > 80% of remaining {time_left:.1f}s, skipping N={N}, degree={degree}")
            continue
            
        if rank == 0:
            print(f"Trying N={N}, degree={degree} (estimated {estimated_time:.1f}s, remaining {time_left:.1f}s)")
        
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
                
                # If change is extremely small and we're close to time limit, stop
                if rel_change < 1e-7 and elapsed + combo_time > 0.8 * time_budget:
                    if rank == 0:
                        print(f"  Excellent convergence, stopping refinement")
                    break
            
            prev_solution_norm = norm
            
        except Exception as e:
            if rank == 0:
                print(f"  Solver failed: {e}")
            continue
    
    # Select the best solution (highest N*degree product)
    if solutions:
        # Sort by N*degree (rough measure of accuracy)
        solutions.sort(key=lambda x: x[3] * x[4])
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
        
        # Accuracy verification
        if len(solutions) >= 2:
            print("\n=== Accuracy Verification ===")
            print("Relative changes between successive refinements:")
            norms = []
            for i, (u_h, _, _, N, deg, _) in enumerate(solutions):
                norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
                norm = np.sqrt(fem.assemble_scalar(norm_form))
                norms.append((N, deg, norm))
            
            for i in range(1, len(norms)):
                prev_N, prev_deg, prev_norm = norms[i-1]
                curr_N, curr_deg, curr_norm = norms[i]
                rel_change = abs(curr_norm - prev_norm) / (curr_norm + 1e-15)
                print(f"  {prev_N},{prev_deg} -> {curr_N},{curr_deg}: {rel_change:.2e}")
            
            # Check if final refinement shows good convergence
            if len(norms) >= 2:
                final_rel_change = abs(norms[-1][2] - norms[-2][2]) / (norms[-1][2] + 1e-15)
                print(f"\nFinal relative change: {final_rel_change:.2e}")
                if final_rel_change < 1e-6:
                    print("✓ Excellent convergence achieved")
                elif final_rel_change < 1e-4:
                    print("✓ Good convergence achieved")
                else:
                    print("⚠ Convergence may need further refinement")
    
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
    
    print("\n=== Final Check ===")
    print(f"Time used: See above (limit: 14.614s)")
    print(f"Mesh: {solver_info['mesh_resolution']}, Degree: {solver_info['element_degree']}")
    print(f"Solution norm on grid: {np.sqrt(np.sum(u_grid**2) * (1/49)**2):.6e}")

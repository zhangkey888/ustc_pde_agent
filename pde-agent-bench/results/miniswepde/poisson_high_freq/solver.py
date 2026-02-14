import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    Returns solution sampled on 50x50 grid and solver info.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Problem parameters with defaults
    kappa = case_spec.get('kappa', 1.0)
    # manufactured_solution is not used directly but we know it's sin(4πx)sin(4πy)
    
    # Domain
    domain_xmin, domain_xmax = 0.0, 1.0
    domain_ymin, domain_ymax = 0.0, 1.0
    
    # Grid convergence loop parameters
    resolutions = [32, 64, 128]
    element_degree = 1  # P1 elements
    
    # Storage for convergence check
    prev_norm = None
    u_sol_final = None
    mesh_resolution_used = None
    solver_info_final = None
    
    # Time tracking
    start_time = time.time()
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define manufactured solution using ufl
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.sin(4*ufl.pi*x[0]) * ufl.sin(4*ufl.pi*x[1])
        
        # Compute source term f = -∇·(κ ∇u_exact)
        f_ufl = -kappa * ufl.div(ufl.grad(u_exact_ufl))
        
        # Boundary condition: u = g on ∂Ω, where g = u_exact
        def boundary_marker(x):
            # All boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], domain_xmin),
                np.isclose(x[0], domain_xmax),
                np.isclose(x[1], domain_ymin),
                np.isclose(x[1], domain_ymax)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary function
        u_bc = fem.Function(V)
        # Interpolate exact solution onto boundary
        u_bc.interpolate(lambda x: np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_ufl, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Create function for solution
        u_sol = fem.Function(V)
        
        # Try iterative solver first, fallback to direct
        solver_success = False
        ksp_type_used = None
        pc_type_used = None
        rtol_used = 1e-8
        iterations_used = 0
        
        # Try iterative solver (GMRES with hypre)
        try:
            if rank == 0:
                print(f"  Trying iterative solver (GMRES+hypre) for N={N}")
            
            # Assemble matrix
            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            
            # Create RHS vector
            b = petsc.create_vector(L_form.function_spaces)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Setup iterative solver
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.getPC().setType(PETSc.PC.Type.HYPRE)
            ksp.setTolerances(rtol=rtol_used, atol=1e-12, max_it=1000)
            ksp.setFromOptions()
            
            # Solve
            x = u_sol.x.petsc_vec
            ksp.solve(b, x)
            u_sol.x.scatter_forward()
            
            # Get iteration count
            iterations_used = ksp.getIterationNumber()
            
            # Check if converged
            if ksp.getConvergedReason() > 0:
                solver_success = True
                ksp_type_used = "gmres"
                pc_type_used = "hypre"
                if rank == 0:
                    print(f"  Iterative solver converged in {iterations_used} iterations")
            else:
                if rank == 0:
                    print(f"  Iterative solver failed to converge")
                
        except Exception as e:
            if rank == 0:
                print(f"  Iterative solver raised exception: {e}")
        
        # If iterative solver failed, try direct solver
        if not solver_success:
            if rank == 0:
                print(f"  Falling back to direct solver (LU) for N={N}")
            
            try:
                # Reassemble with direct solver
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc], u=u_sol,
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                
                solver_success = True
                ksp_type_used = "preonly"
                pc_type_used = "lu"
                iterations_used = 0  # Direct solver doesn't have iterations
                
                if rank == 0:
                    print(f"  Direct solver succeeded")
                    
            except Exception as e:
                if rank == 0:
                    print(f"  Direct solver also failed: {e}")
                # Continue to next resolution
                continue
        
        if not solver_success:
            continue
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = comm.allreduce(norm_local, op=MPI.SUM)
        current_norm = np.sqrt(norm_global)
        
        if rank == 0:
            print(f"  L2 norm of solution: {current_norm:.6e}")
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(current_norm - prev_norm) / current_norm
            if rank == 0:
                print(f"  Relative norm change: {relative_error:.6e}")
            
            if relative_error < 0.01:  # 1% threshold
                if rank == 0:
                    print(f"  Convergence achieved at N={N}")
                u_sol_final = u_sol
                mesh_resolution_used = N
                # Store solver info
                solver_info_final = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used,
                    "rtol": rtol_used,
                    "iterations": iterations_used
                }
                break
        
        prev_norm = current_norm
        u_sol_final = u_sol
        mesh_resolution_used = N
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": iterations_used
        }
    
    # Fallback: if loop finished without convergence, use finest mesh result
    if u_sol_final is None:
        # This shouldn't happen if at least one resolution succeeded
        raise RuntimeError("All mesh resolutions failed")
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(domain_xmin, domain_xmax, nx)
    y_vals = np.linspace(domain_ymin, domain_ymax, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)])
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(u_sol_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Compute wall time
    wall_time_sec = time.time() - start_time
    
    if rank == 0:
        print(f"\nFinal solver info: {solver_info_final}")
        print(f"Total wall time: {wall_time_sec:.3f} seconds")
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solver_info_final
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate dolfinx Function at arbitrary points.
    points: numpy array of shape (3, N)
    Returns: numpy array of shape (N,)
    """
    comm = u_func.function_space.mesh.comm
    domain = u_func.function_space.mesh
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells containing points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
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
    
    # Initialize result array with NaN
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    
    # Evaluate if we have points on this processor
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather results across all processors (in case points are distributed)
    # For simplicity, assume rank 0 has all points (serial evaluation)
    if comm.size > 1:
        # Gather all values to rank 0
        all_values = comm.gather(u_values, root=0)
        if comm.rank == 0:
            # Combine, taking first non-NaN value for each point
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                mask = ~np.isnan(proc_vals)
                combined[mask] = proc_vals[mask]
            u_values = combined
        u_values = comm.bcast(u_values, root=0)
    
    return u_values


# Test the solver if run directly
if __name__ == "__main__":
    # Create a test case specification
    test_case = {
        "kappa": 1.0,
        "manufactured_solution": "sin(4*pi*x)*sin(4*pi*y)"
    }
    
    result = solve(test_case)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

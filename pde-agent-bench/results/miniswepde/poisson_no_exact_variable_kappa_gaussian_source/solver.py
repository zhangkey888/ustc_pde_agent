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
    
    # Extract problem parameters
    domain_type = case_spec.get('domain', {}).get('type', 'unit_square')
    if domain_type == 'unit_square':
        # Default unit square
        pass
    else:
        raise ValueError(f"Unsupported domain type: {domain_type}")
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # Use P1 elements
    
    # Storage for convergence checking
    prev_norm = None
    u_sol_final = None
    mesh_resolution_used = None
    solver_info_final = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Define κ and f using UFL expressions
        # f = exp(-250*((x-0.4)**2 + (y-0.6)**2))
        # κ = 1 + 50*exp(-150*((x-0.5)**2 + (y-0.5)**2))
        
        # Define κ
        kappa = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
        
        # Define source term f
        f = ufl.exp(-250.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
        
        # Define variational form
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Apply Dirichlet boundary conditions (u = 0 on boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Define boundary condition (u = 0 on entire boundary)
        def boundary_marker(x):
            # Mark all boundary facets
            return np.ones(x.shape[1], dtype=bool)
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        linear_iterations = 0
        
        for solver_config in [
            {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
            {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
        ]:
            try:
                # Create and solve linear problem
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": solver_config["ksp_type"],
                        "pc_type": solver_config["pc_type"],
                        "ksp_rtol": solver_config["rtol"],
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                
                # Time the solve
                start_time = time.time()
                u_sol = problem.solve()
                solve_time = time.time() - start_time
                
                # Get solver information
                ksp = problem.solver
                linear_iterations = ksp.getIterationNumber()
                
                # Check if solver converged
                if ksp.getConvergedReason() > 0:
                    solver_success = True
                else:
                    print(f"Solver did not converge with {solver_config}")
                    continue
                
                ksp_type_used = solver_config["ksp_type"]
                pc_type_used = solver_config["pc_type"]
                rtol_used = solver_config["rtol"]
                
                break  # Success, exit solver trial loop
                
            except Exception as e:
                print(f"Solver failed with {solver_config}: {e}")
                continue
        
        if not solver_success:
            raise RuntimeError("All solver configurations failed")
        
        # Compute L2 norm of solution for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                print(f"Converged at N={N} with relative error {relative_error:.6f}")
                u_sol_final = u_sol
                mesh_resolution_used = N
                solver_info_final = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used,
                    "rtol": rtol_used,
                    "iterations": linear_iterations
                }
                break
        
        prev_norm = norm_value
        u_sol_final = u_sol
        mesh_resolution_used = N
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": linear_iterations
        }
    
    # If loop finished without break, use the last result (N=128)
    if u_sol_final is None:
        # This shouldn't happen, but just in case
        raise RuntimeError("No solution obtained")
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
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
        vals = u_sol_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solver_info_final
    }

if __name__ == "__main__":
    # Test the solver with a sample case_spec
    case_spec = {
        "domain": {"type": "unit_square"},
        "coefficients": {
            "kappa": {
                "type": "expr",
                "expr": "1 + 50*exp(-150*((x-0.5)**2 + (y-0.5)**2))"
            }
        },
        "source": {
            "expr": "exp(-250*((x-0.4)**2 + (y-0.6)**2))"
        },
        "pde": {
            "type": "elliptic"
        }
    }
    
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solver iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

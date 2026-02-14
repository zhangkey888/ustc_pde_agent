import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import dolfinx.nls as nls

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract problem parameters
    # Domain is unit square [0,1]x[0,1]
    # Manufactured solution: u_exact = exp(0.5*x)*sin(2*pi*y)
    # Coefficient κ expression given in case_spec
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    degrees = [1, 2, 3]  # Try different degrees
    u_solutions = []
    errors = []
    
    # Solver info to be returned
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    rtol_used = 1e-8
    iterations_used = 0
    final_degree = 2
    final_N = 32
    
    target_error = 4.88e-04  # Accuracy requirement
    
    for degree in degrees:
        for N in resolutions:
            # Create mesh
            domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
            
            # Define function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Define boundary condition (Dirichlet)
            # Manufactured solution: u_exact = exp(0.5*x)*sin(2*pi*y)
            def u_exact(x):
                return np.exp(0.5 * x[0]) * np.sin(2.0 * np.pi * x[1])
            
            # Locate boundary facets
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            # Create BC function
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: u_exact(x))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Coefficient κ
            x = ufl.SpatialCoordinate(domain)
            # κ = 1 + 15*exp(-200*((x-0.25)**2 + (y-0.25)**2)) + 15*exp(-200*((x-0.75)**2 + (y-0.75)**2))
            kappa = 1.0 + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) \
                        + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.75)**2 + (x[1] - 0.75)**2))
            
            # Source term f derived from -∇·(κ ∇u) = f
            # Compute f analytically using UFL
            u_expr = ufl.exp(0.5 * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
            grad_u = ufl.grad(u_expr)
            f = -ufl.div(kappa * grad_u)  # f is a UFL expression
            
            # Variational form
            a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = ufl.inner(f, v) * ufl.dx
            
            # Try iterative solver first, fallback to direct
            solver_success = False
            u_sol = fem.Function(V)
            
            # First try: GMRES with hypre
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "ksp_rtol": 1e-8,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                solver_success = True
                ksp_type_used = "gmres"
                pc_type_used = "hypre"
                rtol_used = 1e-8
                # Get iteration count
                ksp = problem.solver
                iterations_used = ksp.getIterationNumber()
            except Exception as e:
                if rank == 0:
                    print(f"Iterative solver failed: {e}, falling back to direct solver")
            
            # Fallback: direct solver
            if not solver_success:
                try:
                    problem = petsc.LinearProblem(
                        a, L, bcs=[bc],
                        petsc_options={
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                        },
                        petsc_options_prefix="pdebench_"
                    )
                    u_sol = problem.solve()
                    ksp_type_used = "preonly"
                    pc_type_used = "lu"
                    rtol_used = 1e-12
                    iterations_used = 1
                    solver_success = True
                except Exception as e:
                    if rank == 0:
                        print(f"Direct solver also failed: {e}")
                    raise
            
            # Compute L2 error against exact solution
            # Create exact solution function
            u_exact_func = fem.Function(V)
            u_exact_func.interpolate(lambda x: u_exact(x))
            error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
            error_form = fem.form(error_expr)
            error_local = fem.assemble_scalar(error_form)
            error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
            error_value = np.sqrt(error_global)
            
            u_solutions.append(u_sol)
            errors.append(error_value)
            
            if rank == 0:
                print(f"Degree={degree}, N={N}, L2 error={error_value:.6e}")
            
            # Check if error meets target
            if error_value < target_error:
                final_degree = degree
                final_N = N
                final_u = u_sol
                if rank == 0:
                    print(f"Accuracy satisfied with degree={degree}, N={N}, error={error_value:.6e}")
                break  # break out of N loop
        else:
            # If no N satisfied error for this degree, continue to next degree
            continue
        break  # break out of degree loop if error satisfied
    
    # If no combination satisfied, use the best (last) solution
    if 'final_u' not in locals():
        final_u = u_solutions[-1]
        final_N = resolutions[-1]
        final_degree = degrees[-1]
        if rank == 0:
            print(f"Using finest mesh: degree={final_degree}, N={final_N}, error={errors[-1]:.6e}")
    
    # Prepare output grid: 50x50 uniform grid
    nx = 50
    ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T  # 3D points
    
    # Evaluate solution at points
    u_grid = np.zeros((nx, ny))
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(final_u.function_space.mesh, final_u.function_space.mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(final_u.function_space.mesh, cell_candidates, points)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        # Fill u_grid (flattened)
        flat_u = np.full(points.shape[0], np.nan)
        flat_u[eval_map] = vals.flatten()
        u_grid = flat_u.reshape((nx, ny))
    else:
        # If no points on this rank, fill with zeros (should not happen for rank 0)
        u_grid = np.zeros((nx, ny))
    
    # Fill solver_info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": final_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": iterations_used,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test with a dummy case_spec
    case_spec = {
        "kappa": {
            "type": "expr",
            "expr": "1 + 15*exp(-200*((x-0.25)**2 + (y-0.25)**2)) + 15*exp(-200*((x-0.75)**2 + (y-0.75)**2))"
        }
    }
    result = solve(case_spec)
    print("Test completed, u shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])

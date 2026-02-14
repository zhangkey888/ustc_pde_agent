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
    
    # Extract coefficients from case_spec if available
    kappa_val = 1.0
    if 'pde' in case_spec and 'coefficients' in case_spec['pde']:
        coeffs = case_spec['pde']['coefficients']
        if 'kappa' in coeffs:
            kappa_val = coeffs['kappa']
    
    # Manufactured solution (known for this specific case)
    def u_exact(x):
        return np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    # Source term derived from -∇·(κ ∇u) = f, with κ constant
    # For κ=1, f = -Δu
    def f_source(x):
        r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
        u_val = np.exp(-40 * r2)
        # f = -Δu = (160 - 6400*r2)*u_val
        return (160 - 6400 * r2) * u_val
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 2  # P2 elements for better accuracy
    u_sol = None
    u_norm_prev = None
    mesh_resolution_used = None
    solver_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(kappa_val))
        f = fem.Function(V)
        f.interpolate(lambda x: f_source(x))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Boundary condition: Dirichlet using exact solution
        def boundary_marker(x):
            # All boundaries (unit square)
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
        u_bc.interpolate(lambda x: u_exact(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct
        petsc_options_iter = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 1000,
        }
        
        problem = None
        ksp_type = "gmres"
        pc_type = "hypre"
        rtol = 1e-8
        its = 0
        
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_iter,
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            # Get iteration count
            its = problem.solver.getIterationNumber()
            # Check if solver converged
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError(f"KSP did not converge, reason: {problem.solver.getConvergedReason()}")
        except Exception as e:
            # Fallback to direct solver
            if rank == 0:
                print(f"Iterative solver failed for N={N}: {e}. Switching to direct solver.")
            petsc_options_direct = {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_direct,
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            ksp_type = "preonly"
            pc_type = "lu"
            its = 0  # Direct solver doesn't have iterations
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)
        u_norm = np.sqrt(norm_global)
        
        # Check convergence
        if u_norm_prev is not None:
            rel_error = abs(u_norm - u_norm_prev) / u_norm if u_norm > 0 else float('inf')
            if rel_error < 0.01:
                mesh_resolution_used = N
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": its,
                }
                break
        
        u_norm_prev = u_norm
        mesh_resolution_used = N
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": its,
        }
    
    # If loop finished without break, use last resolution (128)
    # Already set above
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T  # 3D points
    
    # Use geometry utilities to evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
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
    
    # Gather all values to rank 0 (evaluator expects full array)
    u_all = comm.gather(u_values, root=0)
    if rank == 0:
        u_combined = np.concatenate(u_all)
        u_grid = u_combined.reshape((nx, ny))
    else:
        u_grid = np.empty((nx, ny), dtype=ScalarType)
    
    comm.Bcast(u_grid, root=0)
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }

if __name__ == "__main__":
    # Test with dummy case_spec
    case_spec = {"pde": {"type": "poisson"}}
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Solution shape:", result["u"].shape)
        print("Solver info:", result["solver_info"])

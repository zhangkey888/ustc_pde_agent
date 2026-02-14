import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Problem parameters
    kappa = 5.0
    # Manufactured solution u_exact = cos(2πx)cos(3πy)
    # Source term f = -∇·(κ∇u) = 13π² κ cos(2πx)cos(3πy)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    solver_info = {}
    domain = None
    petsc_options_used = {"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Dirichlet BC: u = u_exact on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Interpolate exact solution for BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.cos(2*np.pi*x[0]) * np.cos(3*np.pi*x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define source term f as a fem.Function
        f = fem.Function(V)
        f.interpolate(lambda x: 13.0 * (np.pi**2) * kappa * np.cos(2*np.pi*x[0]) * np.cos(3*np.pi*x[1]))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        petsc_options = {"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8}
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], petsc_options=petsc_options,
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            # Check if solver converged
            ksp = problem.solver
            its = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(f"KSP diverged with reason {reason}")
            solver_info["iterations"] = its
            petsc_options_used = petsc_options.copy()
        except Exception as e:
            # Fallback to direct solver
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], petsc_options=petsc_options,
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            ksp = problem.solver
            its = ksp.getIterationNumber()
            solver_info["iterations"] = its
            petsc_options_used = petsc_options.copy()
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_val = comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM)
        norm_val = np.sqrt(norm_val)
        
        # Check convergence
        if norm_old is not None:
            rel_error = abs(norm_val - norm_old) / norm_val if norm_val != 0 else 0.0
            if rel_error < 0.01:
                # Converged
                solver_info.update({
                    "mesh_resolution": N,
                    "element_degree": 1,
                    "ksp_type": petsc_options_used.get("ksp_type", "gmres"),
                    "pc_type": petsc_options_used.get("pc_type", "hypre"),
                    "rtol": petsc_options_used.get("ksp_rtol", 1e-8),
                })
                break
        norm_old = norm_val
    
    # If loop finished without break, use the last resolution (128)
    if "mesh_resolution" not in solver_info:
        solver_info.update({
            "mesh_resolution": 128,
            "element_degree": 1,
            "ksp_type": petsc_options_used.get("ksp_type", "gmres"),
            "pc_type": petsc_options_used.get("pc_type", "hypre"),
            "rtol": petsc_options_used.get("ksp_rtol", 1e-8),
        })
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T  # shape (N, 3)
    
    # Find cells containing points
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    # Gather all values to root if running in parallel
    if comm.size > 1:
        # Simple gather to rank 0 (evaluator runs sequentially)
        all_values = comm.gather(u_values, root=0)
        if comm.rank == 0:
            # Combine: take first non-nan value for each point
            combined = np.full_like(u_values, np.nan)
            for arr in all_values:
                mask = ~np.isnan(arr)
                combined[mask] = arr[mask]
            u_values = combined
        else:
            u_values = np.empty_like(u_values)
        u_values = comm.bcast(u_values, root=0)
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test with a dummy case_spec
    case_spec = {"pde": {"type": "elliptic"}}
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)

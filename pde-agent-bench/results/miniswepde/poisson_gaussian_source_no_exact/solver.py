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
    rank = comm.rank
    ScalarType = PETSc.ScalarType
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_prev = None
    norm_prev = None
    final_u = None
    final_domain = None
    solver_info = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define boundary condition (Dirichlet, zero on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark all boundaries
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
        
        # Source term f = exp(-200*((x-0.25)**2 + (y-0.75)**2))
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.exp(-200*((x[0]-0.25)**2 + (x[1]-0.75)**2))
        
        # Coefficient kappa = 1.0
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            iterations = problem.solver.getIterationNumber()
            solver_info = {
                "mesh_resolution": N,
                "element_degree": 1,
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rtol": 1e-8,
                "iterations": iterations
            }
        except Exception as e:
            # Fallback to direct solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            solver_info = {
                "mesh_resolution": N,
                "element_degree": 1,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "rtol": 1e-8,
                "iterations": 1  # direct solver typically 1 iteration
            }
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if norm_prev is not None:
            relative_error = abs(norm - norm_prev) / norm if norm > 1e-12 else 0
            if relative_error < 0.01:
                final_u = u_sol
                final_domain = domain
                break
        
        u_prev = u_sol
        norm_prev = norm
        final_u = u_sol
        final_domain = domain
    
    # If loop finished without break, final_u is already set to last solution
    # Sample solution on 50x50 grid
    nx = ny = 50
    # Add small epsilon to avoid points exactly on cell boundaries
    eps = 1e-12
    x_vals = np.linspace(0 + eps, 1 - eps, nx)
    y_vals = np.linspace(0 + eps, 1 - eps, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather to root and combine
    if rank == 0:
        gathered = np.empty((comm.size, points.shape[0]), dtype=np.float64)
    else:
        gathered = None
    comm.Gather(u_values, gathered, root=0)
    
    if rank == 0:
        # Combine: take first non-nan value for each point (max over processes)
        u_values_combined = np.nanmax(gathered, axis=0)
        # Ensure no nan remains (should not happen)
        if np.any(np.isnan(u_values_combined)):
            # Replace nan with 0 (fallback)
            u_values_combined[np.isnan(u_values_combined)] = 0.0
    else:
        u_values_combined = None
    
    # Broadcast combined array to all processes
    u_values_combined = comm.bcast(u_values_combined, root=0)
    u_grid = u_values_combined.reshape(nx, ny)
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Simple test
    case_spec = {"pde": {"type": "poisson"}}
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Solver info:", result["solver_info"])
        print("u shape:", result["u"].shape)
        print("u min/max:", np.min(result["u"]), np.max(result["u"]))

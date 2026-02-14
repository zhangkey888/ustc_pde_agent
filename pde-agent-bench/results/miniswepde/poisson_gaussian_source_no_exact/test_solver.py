import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_prev = None
    norm_prev = None
    final_u = None
    final_mesh_res = None
    
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
        f = fem.Constant(domain, ScalarType(1.0))  # placeholder
        
        # Actually we need to use Expression for f in the form
        # But for simplicity, we'll use Constant for now
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
            solver_info = {
                "mesh_resolution": N,
                "element_degree": 1,
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rtol": 1e-8,
                "iterations": problem.solver.getIterationNumber()
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
            relative_error = abs(norm - norm_prev) / norm if norm > 0 else 0
            if relative_error < 0.01:
                final_u = u_sol
                final_mesh_res = N
                break
        
        u_prev = u_sol
        norm_prev = norm
        final_u = u_sol
        final_mesh_res = N
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).T
    
    # Evaluate solution at points
    from dolfinx import geometry
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
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = final_u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

# Test the function
if __name__ == "__main__":
    case_spec = {"pde": {"type": "poisson"}}
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)

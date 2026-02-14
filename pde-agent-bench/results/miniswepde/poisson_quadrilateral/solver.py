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
    
    # Problem parameters
    kappa = case_spec.get('kappa', 2.0)
    
    # Manufactured solution
    def u_exact_func(x):
        return np.exp(x[0]) * np.cos(2.0 * np.pi * x[1])
    
    # Try configurations in order of increasing cost, stop when accuracy met
    # Prefer quadratic elements with moderate mesh
    configs = [
        (64, 2),   # Likely sufficient
        (128, 1),  # Fallback
        (128, 2),  # More accurate
        (256, 1),  # Last resort
    ]
    
    u_sol = None
    domain = None
    solver_info = {}
    
    for N, element_degree in configs:
        start_time = time.time()
        
        # Create mesh with quadrilateral cells
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term f = -∇·(κ ∇u_exact)
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.exp(x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
        f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))
        
        # Bilinear and linear forms
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_ufl, v) * ufl.dx
        
        # Boundary conditions: Dirichlet using exact solution on all boundaries
        # Create facet to cell connectivity
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        
        # Mark boundary facets
        def boundary_marker(x):
            # Return True for points on boundary
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        
        # Locate DOFs on boundary
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition
        u_bc = fem.Function(V)
        u_bc.interpolate(u_exact_func)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct
        solver_success = False
        linear_iterations = 0
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        
        for solver_config in [('gmres', 'hypre'), ('preonly', 'lu')]:
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": solver_config[0],
                        "pc_type": solver_config[1],
                        "ksp_rtol": rtol,
                        "ksp_atol": 1e-12,
                        "ksp_max_it": 1000
                    },
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                
                # Get iteration count
                ksp = problem._solver
                linear_iterations = ksp.getIterationNumber()
                ksp_type = solver_config[0]
                pc_type = solver_config[1]
                solver_success = True
                break
            except Exception as e:
                if solver_config[0] == 'preonly':  # Last resort failed
                    raise RuntimeError(f"All solvers failed: {e}")
                continue
        
        if not solver_success:
            raise RuntimeError("Solver failed for all configurations")
        
        # Compute L2 error against exact solution
        error_expr = ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx
        error_form = fem.form(error_expr)
        error = fem.assemble_scalar(error_form)
        error = np.sqrt(comm.allreduce(error, op=MPI.SUM))
        
        solve_time = time.time() - start_time
        
        # Check if accuracy requirement is met
        if error <= 8.70e-05:
            solver_info = {
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": linear_iterations
            }
            break
    
    # If no configuration met accuracy (should not happen), use last one
    if not solver_info:
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": linear_iterations
        }
    
    # Evaluate solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T
    
    # Probe points
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
    
    # Gather all values on root process (use SUM since NaN + value = NaN)
    u_all = np.zeros(nx * ny, dtype=ScalarType)
    comm.Allreduce(u_values, u_all, op=MPI.SUM)
    u_grid = u_all.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a simple case specification
    case_spec = {"kappa": 2.0}
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

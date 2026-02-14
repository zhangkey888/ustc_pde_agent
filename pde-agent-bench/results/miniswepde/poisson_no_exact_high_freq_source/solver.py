import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing PDE specification. Expected keys:
        - "pde": dict with "coefficients" subdict containing "kappa" (diffusion coefficient)
    
    Returns:
    --------
    dict with keys:
        - "u": numpy array of shape (50, 50) with solution values on uniform grid
        - "solver_info": dict with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract problem parameters from case_spec
    kappa = case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0)
    
    # Source term: f = sin(12*pi*x)*sin(10*pi*y) (given in problem description)
    # Domain: unit square [0,1]x[0,1]
    # Boundary condition: u = 0 on entire boundary (Dirichlet) - standard for this problem
    
    # Adaptive mesh refinement parameters
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    rtol = 1e-8  # Linear solver tolerance
    
    # Storage for convergence check
    u_prev = None
    norm_prev = None
    solution_info = {}
    u_sol = None
    domain = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (u = 0 on entire boundary)
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
        
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.sin(12 * np.pi * x[0]) * ufl.sin(10 * np.pi * x[1])
        
        # Weak form: ∫(kappa * ∇u·∇v) dx = ∫(f * v) dx
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        solver_succeeded = False
        linear_iterations = 0
        ksp_type_used = "gmres"
        pc_type_used = "hypre"
        
        # First try: iterative solver (GMRES with hypre)
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-12,
                    "ksp_max_it": 1000
                },
                petsc_options_prefix="pde_"
            )
            u_sol = problem.solve()
            linear_iterations = problem.solver.getIterationNumber()
            solver_succeeded = True
        except Exception as e:
            print(f"Iterative solver failed at N={N}: {e}")
            solver_succeeded = False
        
        # Fallback: direct solver (LU)
        if not solver_succeeded:
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "ksp_rtol": rtol,
                        "ksp_atol": 1e-12
                    },
                    petsc_options_prefix="pde_"
                )
                u_sol = problem.solve()
                linear_iterations = problem.solver.getIterationNumber()
                ksp_type_used = "preonly"
                pc_type_used = "lu"
                solver_succeeded = True
            except Exception as e:
                print(f"Direct solver also failed at N={N}: {e}")
                raise
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_prev is not None:
            relative_error = abs(norm_value - norm_prev) / norm_value if norm_value > 0 else 0.0
            if relative_error < 0.01:  # 1% convergence criterion
                print(f"Converged at resolution N={N} with relative error {relative_error:.6f}")
                solution_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used,
                    "rtol": rtol,
                    "iterations": linear_iterations
                }
                break
        
        norm_prev = norm_value
        u_prev = u_sol
        
        # If we reach the last resolution, use it
        if N == resolutions[-1]:
            print(f"Using finest resolution N={N} (convergence not reached)")
            solution_info = {
                "mesh_resolution": N,
                "element_degree": element_degree,
                "ksp_type": ksp_type_used,
                "pc_type": pc_type_used,
                "rtol": rtol,
                "iterations": linear_iterations
            }
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    from dolfinx import geometry
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": solution_info
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver iterations: {result['solver_info']['iterations']}")

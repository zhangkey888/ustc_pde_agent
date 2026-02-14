import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Manufactured solution
    def u_exact(x):
        return np.exp(5 * x[0]) * np.sin(np.pi * x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]  # Progressive refinement
    u_sol = None
    norm_old = None
    mesh_resolution_used = None
    element_degree = 2  # Polynomial degree
    total_iterations = 0
    ksp_type = "cg"
    pc_type = "hypre"
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: u_exact(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Source term f = -∇·(∇u_exact) with κ=1
        x = ufl.SpatialCoordinate(domain)
        u_expr = ufl.exp(5 * x[0]) * ufl.sin(np.pi * x[1])
        f_expr = -ufl.div(ufl.grad(u_expr))
        
        # Variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        κ = fem.Constant(domain, ScalarType(1.0))
        
        # Interpolate f_expr onto a function
        f_func = fem.Function(V)
        f_expr_compiled = fem.Expression(f_expr, V.element.interpolation_points)
        f_func.interpolate(f_expr_compiled)
        
        a = ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Solve linear problem with iterative solver first, fallback to direct
        petsc_options = {
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": 1e-8,
            "ksp_atol": 1e-10,
            "ksp_max_it": 1000,
        }
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix="pde_"
            )
            u_sol = problem.solve()
            # Get iteration count
            ksp = problem._solver
            total_iterations += ksp.getIterationNumber()
        except Exception as e:
            # Fallback to direct solver
            petsc_options_fallback = {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options_fallback,
                petsc_options_prefix="pde_"
            )
            u_sol = problem.solve()
            ksp = problem._solver
            total_iterations += ksp.getIterationNumber()
            ksp_type = "preonly"
            pc_type = "lu"
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence: relative change in norm < 0.01 (1%)
        if norm_old is not None:
            rel_error = abs(norm_value - norm_old) / norm_value
            if rel_error < 0.01:
                mesh_resolution_used = N
                break
        norm_old = norm_value
        mesh_resolution_used = N
    
    # If loop finished without break, use last resolution
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
    
    # Evaluate solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T  # 3D points
    
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
    
    u_grid_flat = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": total_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {}
    import time
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    print("Time:", end - start)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters - use degree 3 elements with finer mesh to maximize accuracy within time budget
    N = 80
    degree = 3
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.exp(2 * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Source term: -div(kappa * grad(u_exact)) = f
    # u = exp(2x)*cos(pi*y)
    # u_xx = 4*exp(2x)*cos(pi*y)
    # u_yy = -pi^2*exp(2x)*cos(pi*y)
    # laplacian = (4 - pi^2)*exp(2x)*cos(pi*y)
    # f = -kappa * laplacian = (pi^2 - 4)*exp(2x)*cos(pi*y)
    f_expr = (ufl.pi**2 - 4.0) * ufl.exp(2 * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Boundary condition
    u_bc_func = fem.Function(V)
    u_exact_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_fem_expr)
    
    # Mark all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-15",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get actual iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_fem_expr)
    error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    print(f"Mesh: {N}, Degree: {degree}, Iterations: {iterations}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {np.nanmin(result['u']):.6f}, max: {np.nanmax(result['u']):.6f}")
    
    # Check against exact solution on grid
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact_grid = np.exp(2 * XX) * np.cos(np.pi * YY)
    grid_error = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    print(f"Grid RMSE: {grid_error:.6e}")
    max_error = np.nanmax(np.abs(result['u'] - u_exact_grid))
    print(f"Grid max error: {max_error:.6e}")

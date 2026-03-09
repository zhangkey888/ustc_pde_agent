import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation with multi-frequency source."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec for any overrides
    nx_eval = 50
    ny_eval = 50
    
    # For this problem with high-frequency source terms (up to 9*pi*x, 7*pi*y),
    # we need sufficient mesh resolution. Let's use adaptive approach.
    # The highest frequency is 9*pi ≈ 28.3, so we need at least ~30 points per direction
    # for degree 1. With degree 2, we can use fewer elements.
    
    # Use degree 2 elements for better accuracy with fewer DOFs
    element_degree = 2
    N = 64  # mesh resolution
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    f_expr = ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) + \
             0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1])
    
    # kappa = 1.0
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Bilinear form: a(u,v) = kappa * inner(grad(u), grad(v)) * dx
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # Linear form: L(v) = f * v * dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries (g=0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,  # not easily accessible from LinearProblem
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {result['u'].min():.6e}, max: {result['u'].max():.6e}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
    
    # Quick sanity check: for the Poisson equation -laplacian(u) = f with homogeneous BCs,
    # the exact solution for sin(m*pi*x)*sin(n*pi*y) source is:
    # u = sin(m*pi*x)*sin(n*pi*y) / (pi^2*(m^2+n^2))
    # So for our source:
    # u_exact = sin(5*pi*x)*sin(3*pi*y)/(pi^2*(25+9)) + 0.5*sin(9*pi*x)*sin(7*pi*y)/(pi^2*(81+49))
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    u_exact = (np.sin(5*np.pi*X)*np.sin(3*np.pi*Y) / (np.pi**2 * 34) +
               0.5*np.sin(9*np.pi*X)*np.sin(7*np.pi*Y) / (np.pi**2 * 130))
    
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    print(f"RMS error vs analytical: {error:.6e}")
    print(f"Max abs error: {np.max(np.abs(result['u'] - u_exact)):.6e}")

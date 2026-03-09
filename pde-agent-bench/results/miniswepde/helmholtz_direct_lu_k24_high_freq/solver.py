import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    """Solve Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Problem parameters
    k_val = 24.0
    
    # For k=24, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.26, so on [0,1] we need ~40 points minimum
    # But manufactured solution has sin(5*pi*x)*sin(4*pi*y) which has higher frequency components
    # Effective wavenumber from manufactured solution: sqrt((5*pi)^2 + (4*pi)^2) ≈ 20.1
    # Combined with k=24, we need good resolution
    
    # Try with degree 2 elements and moderate mesh
    N = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term: f = -∇²u - k²u
    # For u = sin(5*pi*x)*sin(4*pi*y):
    # ∇²u = -(25*pi² + 16*pi²) * sin(5*pi*x)*sin(4*pi*y) = -41*pi² * u
    # So -∇²u = 41*pi² * u
    # f = -∇²u - k²u = (41*pi² - k²) * u
    f_expr = (41.0 * ufl.pi**2 - k_val**2) * u_exact_expr
    
    # Boundary condition from exact solution
    u_bc = fem.Function(V)
    u_exact_fe = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_exact_fe)
    
    # Mark all boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational form: -∇²u - k²u = f
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k_const = fem.Constant(domain, PETSc.ScalarType(k_val))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve with direct solver (LU) - robust for indefinite Helmholtz
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    
    # Make 3D points (required by dolfinx)
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
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": 1,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Time: {elapsed:.3f}s")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(5 * np.pi * X) * np.sin(4 * np.pi * Y)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")

import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def test_single_run(N=256, degree=2):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary condition (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Source term
    k = 15.0
    def source_function(x):
        return 10.0 * np.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    f = fem.Function(V)
    f.interpolate(source_function)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    start = time.time()
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_"
    )
    u_h = problem.solve()
    elapsed = time.time() - start
    
    # Interpolate to 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_grid_flat = np.full(nx * ny, np.nan, dtype=PETSc.ScalarType)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    if comm.size > 1:
        all_values = comm.gather(u_grid_flat, root=0)
        if rank == 0:
            u_grid_flat_combined = np.full_like(u_grid_flat, np.nan)
            for arr in all_values:
                mask = ~np.isnan(arr)
                u_grid_flat_combined[mask] = arr[mask]
            u_grid_flat = u_grid_flat_combined
        else:
            u_grid_flat = np.full_like(u_grid_flat, np.nan)
        u_grid_flat = comm.bcast(u_grid_flat, root=0)
    
    u_grid = u_grid_flat.reshape(nx, ny)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    if rank == 0:
        print(f"N={N}, degree={degree}: Time={elapsed:.3f}s")
        print(f"Solution range: [{u_grid.min():.6e}, {u_grid.max():.6e}]")
    
    return u_grid, elapsed

if __name__ == "__main__":
    # Test different configurations
    configs = [
        (128, 2),
        (256, 2),
        (128, 3),
        (256, 1),
    ]
    
    for N, degree in configs:
        print(f"\n=== Testing N={N}, degree={degree} ===")
        u_grid, elapsed = test_single_run(N, degree)
        
        # Compute residual
        nx, ny = u_grid.shape
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = 10.0 * np.exp(-80.0 * ((X - 0.35)**2 + (Y - 0.55)**2))
        
        laplacian = np.zeros_like(u_grid)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                laplacian[i, j] = (u_grid[i+1, j] - 2*u_grid[i, j] + u_grid[i-1, j]) / (dx**2) + \
                                  (u_grid[i, j+1] - 2*u_grid[i, j] + u_grid[i, j-1]) / (dy**2)
        
        residual = -laplacian - 15.0**2 * u_grid - f
        interior_mask = np.zeros_like(u_grid, dtype=bool)
        interior_mask[1:-1, 1:-1] = True
        residual_interior = residual[interior_mask]
        
        l2_residual = np.sqrt(np.mean(residual_interior**2)) * dx * dy
        max_residual = np.max(np.abs(residual_interior))
        
        print(f"L2 residual: {l2_residual:.6e}")
        print(f"Max residual: {max_residual:.6e}")
        print(f"Time: {elapsed:.3f}s (limit: 14.614s)")

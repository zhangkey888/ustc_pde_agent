import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
from dolfinx import geometry

def solve_helmholtz(N, degree, k=15.0):
    """Solve Helmholtz equation with given mesh resolution and degree."""
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
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_"
    )
    u_h = problem.solve()
    
    return u_h, domain, V

def interpolate_to_grid(u_h, domain, nx=50, ny=50):
    """Interpolate solution to uniform grid."""
    comm = domain.comm
    rank = comm.rank
    
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
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
    
    return u_grid

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Compute reference solution (very fine)
    if rank == 0:
        print("Computing reference solution (N=512, degree=3)...")
    start_ref = time.time()
    u_ref, domain_ref, V_ref = solve_helmholtz(256, 3)
    u_ref_grid = interpolate_to_grid(u_ref, domain_ref)
    time_ref = time.time() - start_ref
    
    # Compute current solver solution (N=128, degree=2)
    if rank == 0:
        print("Computing current solver solution (N=128, degree=2)...")
    start_curr = time.time()
    u_curr, domain_curr, V_curr = solve_helmholtz(128, 2)
    u_curr_grid = interpolate_to_grid(u_curr, domain_curr)
    time_curr = time.time() - start_curr
    
    if rank == 0:
        print(f"\nReference solution time: {time_ref:.3f}s")
        print(f"Current solution time: {time_curr:.3f}s")
        
        # Compute L2 error between solutions
        nx, ny = u_ref_grid.shape
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        
        error = u_curr_grid - u_ref_grid
        l2_error = np.sqrt(np.sum(error**2) * dx * dy)
        max_error = np.max(np.abs(error))
        
        print(f"\nL2 error (vs reference): {l2_error:.6e}")
        print(f"Max error (vs reference): {max_error:.6e}")
        print(f"Accuracy requirement: ≤ 1.63e-01")
        
        # Also compute residual for current solution
        laplacian = np.zeros_like(u_curr_grid)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                laplacian[i, j] = (u_curr_grid[i+1, j] - 2*u_curr_grid[i, j] + u_curr_grid[i-1, j]) / (dx**2) + \
                                  (u_curr_grid[i, j+1] - 2*u_curr_grid[i, j] + u_curr_grid[i, j-1]) / (dy**2)
        
        X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
        f = 10.0 * np.exp(-80.0 * ((X - 0.35)**2 + (Y - 0.55)**2))
        residual = -laplacian - 15.0**2 * u_curr_grid - f
        
        interior_mask = np.zeros_like(u_curr_grid, dtype=bool)
        interior_mask[1:-1, 1:-1] = True
        residual_interior = residual[interior_mask]
        
        l2_residual = np.sqrt(np.mean(residual_interior**2)) * dx * dy
        max_residual = np.max(np.abs(residual_interior))
        
        print(f"\nL2 residual (current): {l2_residual:.6e}")
        print(f"Max residual (current): {max_residual:.6e}")

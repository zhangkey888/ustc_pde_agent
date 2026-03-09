import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_at_resolution(N, element_degree=2):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    kappa = 0.2 + 0.8 * ufl.exp(-80.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
        petsc_options_prefix=f"p{N}_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
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
    
    return u_values.reshape((nx_out, ny_out))

# Convergence study
resolutions = [32, 48, 64, 96, 128]
prev_grid = None
for N in resolutions:
    t0 = time.time()
    u_grid = solve_at_resolution(N, element_degree=2)
    elapsed = time.time() - t0
    
    if prev_grid is not None:
        diff = np.nanmax(np.abs(u_grid - prev_grid))
        rel_diff = diff / (np.nanmax(np.abs(u_grid)) + 1e-15)
        print(f"N={N:4d}, time={elapsed:.3f}s, max={np.nanmax(u_grid):.8f}, max_diff={diff:.2e}, rel_diff={rel_diff:.2e}")
    else:
        print(f"N={N:4d}, time={elapsed:.3f}s, max={np.nanmax(u_grid):.8f}")
    
    prev_grid = u_grid.copy()

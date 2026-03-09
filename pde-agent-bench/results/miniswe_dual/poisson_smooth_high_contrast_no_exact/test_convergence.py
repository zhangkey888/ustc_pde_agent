import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_at_resolution(N, degree):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    
    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5)**2)
    pi = ufl.pi
    f = 1.0 + ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_arr: np.zeros_like(x_arr[0]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
        petsc_options_prefix=f"p{N}_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx_out, ny_out))

# Test convergence
resolutions = [32, 64, 128, 256]
degrees = [1, 2]
prev = {}

for deg in degrees:
    prev_grid = None
    for N in resolutions:
        t0 = time.time()
        grid = solve_at_resolution(N, deg)
        elapsed = time.time() - t0
        
        if prev_grid is not None:
            diff = np.max(np.abs(grid - prev_grid))
            rel_diff = diff / (np.max(np.abs(grid)) + 1e-15)
            print(f"P{deg}, N={N}: max={np.max(grid):.8f}, max_diff={diff:.2e}, rel_diff={rel_diff:.2e}, time={elapsed:.3f}s")
        else:
            print(f"P{deg}, N={N}: max={np.max(grid):.8f}, time={elapsed:.3f}s")
        
        prev_grid = grid.copy()

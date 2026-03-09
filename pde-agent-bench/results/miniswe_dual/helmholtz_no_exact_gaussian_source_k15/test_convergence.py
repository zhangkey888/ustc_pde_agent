import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve_helmholtz(N, degree=2):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    k = fem.Constant(domain, PETSc.ScalarType(15.0))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"helm_{N}_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    points_3d = np.zeros((nx_out*ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.zeros(points_3d.shape[0])
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
prev = None
for N in [32, 64, 80, 128, 160]:
    t0 = time.time()
    u_grid = solve_helmholtz(N, degree=2)
    elapsed = time.time() - t0
    norm = np.linalg.norm(u_grid)
    if prev is not None:
        diff = np.linalg.norm(u_grid - prev) / (norm + 1e-15)
        print(f"N={N:4d}, degree=2, norm={norm:.8f}, rel_diff={diff:.2e}, max={u_grid.max():.8f}, min={u_grid.min():.8f}, time={elapsed:.3f}s")
    else:
        print(f"N={N:4d}, degree=2, norm={norm:.8f}, max={u_grid.max():.8f}, min={u_grid.min():.8f}, time={elapsed:.3f}s")
    prev = u_grid.copy()

# Also test P3
print("\n--- P3 elements ---")
prev3 = None
for N in [32, 64, 80]:
    t0 = time.time()
    u_grid = solve_helmholtz(N, degree=3)
    elapsed = time.time() - t0
    norm = np.linalg.norm(u_grid)
    if prev3 is not None:
        diff = np.linalg.norm(u_grid - prev3) / (norm + 1e-15)
        print(f"N={N:4d}, degree=3, norm={norm:.8f}, rel_diff={diff:.2e}, max={u_grid.max():.8f}, min={u_grid.min():.8f}, time={elapsed:.3f}s")
    else:
        print(f"N={N:4d}, degree=3, norm={norm:.8f}, max={u_grid.max():.8f}, min={u_grid.min():.8f}, time={elapsed:.3f}s")
    prev3 = u_grid.copy()

# Compare P2 N=128 vs P3 N=80
u_p2_128 = solve_helmholtz(128, degree=2)
u_p3_80 = solve_helmholtz(80, degree=3)
diff = np.linalg.norm(u_p2_128 - u_p3_80) / (np.linalg.norm(u_p2_128) + 1e-15)
print(f"\nP2 N=128 vs P3 N=80: rel_diff = {diff:.2e}")

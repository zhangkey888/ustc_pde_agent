import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_at_resolution(N, degree=2):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    
    kappa = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    f = ufl.exp(-250.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = f * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    problem = petsc.LinearProblem(
        a, L_form, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
        petsc_options_prefix=f"conv_{N}_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
resolutions = [32, 64, 96, 128, 192]
results = {}
for N in resolutions:
    t0 = time.time()
    u_grid = solve_at_resolution(N, degree=2)
    elapsed = time.time() - t0
    results[N] = u_grid
    print(f"N={N:4d}: time={elapsed:.3f}s, max={np.nanmax(u_grid):.8e}, min={np.nanmin(u_grid):.8e}")

# Compare consecutive resolutions
print("\nConvergence analysis (max absolute difference on 50x50 grid):")
prev_N = None
for N in resolutions:
    if prev_N is not None:
        diff = np.abs(results[N] - results[prev_N])
        max_diff = np.nanmax(diff)
        l2_diff = np.sqrt(np.nanmean(diff**2))
        print(f"  N={prev_N} -> N={N}: max_diff={max_diff:.6e}, l2_diff={l2_diff:.6e}")
    prev_N = N

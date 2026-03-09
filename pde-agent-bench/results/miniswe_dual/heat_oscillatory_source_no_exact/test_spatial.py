import numpy as np
import time
import sys

# Modify solver to accept custom N
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve_custom(N, dt_val=0.02, element_degree=1):
    kappa_val = 0.8
    t_end = 0.12
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt = fem.Constant(domain, ScalarType(dt_val))
    
    a = (u / dt) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt) * v * ufl.dx + f * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    n_steps = int(np.round(t_end / dt_val))
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="heat_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"},
    )
    
    for step in range(n_steps):
        u_h = problem.solve()
        u_n.x.array[:] = u_h.x.array[:]
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx_out, ny_out))

# Test different resolutions
for N in [32, 64, 128]:
    for deg in [1, 2]:
        t0 = time.time()
        u = solve_custom(N, dt_val=0.005, element_degree=deg)
        elapsed = time.time() - t0
        print(f"N={N:3d}, deg={deg}: time={elapsed:.3f}s, max={np.nanmax(u):.8f}, min={np.nanmin(u):.8f}")

# Compare N=64 vs N=128 (both P1)
u64 = solve_custom(64, dt_val=0.005, element_degree=1)
u128 = solve_custom(128, dt_val=0.005, element_degree=1)
diff = np.abs(u64 - u128)
print(f"\nN=64 vs N=128 (P1): max_diff={np.nanmax(diff):.8e}, rel_diff={np.nanmax(diff)/np.nanmax(np.abs(u128)):.6e}")

# Compare P1 vs P2 at N=64
u64p1 = solve_custom(64, dt_val=0.005, element_degree=1)
u64p2 = solve_custom(64, dt_val=0.005, element_degree=2)
diff2 = np.abs(u64p1 - u64p2)
print(f"N=64 P1 vs P2: max_diff={np.nanmax(diff2):.8e}, rel_diff={np.nanmax(diff2)/np.nanmax(np.abs(u64p2)):.6e}")

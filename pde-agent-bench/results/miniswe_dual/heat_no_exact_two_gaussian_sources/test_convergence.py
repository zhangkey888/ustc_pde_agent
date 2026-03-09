import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_heat(N, degree, dt_val, t_end=0.1):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n.x.array[:] = 0.0
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-220.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
        ufl.exp(-220.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2))
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = ufl.inner(u, v) / dt_c * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) / dt_c * ufl.dx + ufl.inner(f, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    solver.setUp()
    
    t = 0.0
    n_steps = 0
    while t < t_end - 1e-14:
        t += dt_val
        n_steps += 1
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    
    return u_grid, n_steps

# Test different configurations
configs = [
    (32, 1, 0.02),
    (64, 1, 0.02),
    (128, 1, 0.02),
    (64, 2, 0.02),
    (64, 1, 0.005),
    (128, 1, 0.005),
]

results = {}
for N, deg, dt in configs:
    t0 = time.time()
    u_grid, nsteps = solve_heat(N, deg, dt)
    elapsed = time.time() - t0
    maxval = np.nanmax(u_grid)
    l2 = np.sqrt(np.nanmean(u_grid**2))
    print(f"N={N:3d}, deg={deg}, dt={dt:.4f}, steps={nsteps}, max={maxval:.6f}, L2={l2:.6f}, time={elapsed:.3f}s")
    results[(N, deg, dt)] = u_grid

# Compare solutions
ref = results[(128, 1, 0.005)]
for key, u_grid in results.items():
    diff = np.abs(u_grid - ref)
    max_diff = np.nanmax(diff)
    l2_diff = np.sqrt(np.nanmean(diff**2))
    print(f"Config {key} vs ref: max_diff={max_diff:.6e}, L2_diff={l2_diff:.6e}")

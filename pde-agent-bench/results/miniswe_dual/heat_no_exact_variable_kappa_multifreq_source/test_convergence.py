import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_heat(N, dt, degree=1):
    t_end = 0.1
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    kappa_expr = 1.0 + 0.6 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f_expr = (ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1]) 
              + 0.3 * ufl.sin(10 * pi * x[0]) * ufl.sin(9 * pi * x[1]))
    
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = (ufl.inner(u, v) / dt_const * ufl.dx 
         + ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (ufl.inner(f_expr, v) * ufl.dx 
         + ufl.inner(u_n, v) / dt_const * ufl.dx)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b_func = fem.Function(V)
    b_vec = b_func.x.petsc_vec
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    solver.setUp()
    
    num_steps = int(round(t_end / dt))
    for step in range(num_steps):
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
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    solver.destroy()
    A.destroy()
    
    return u_values.reshape((nx_out, ny_out))

# Reference: N=128, dt=0.002, P2
t0 = time.time()
ref = solve_heat(128, 0.002, degree=2)
t1 = time.time()
print(f"Reference (N=128, dt=0.002, P2): {t1-t0:.2f}s, range=[{ref.min():.6f}, {ref.max():.6f}]")

# Test our default config
t0 = time.time()
sol = solve_heat(64, 0.02, 1)
t1 = time.time()
err = np.max(np.abs(sol - ref))
l2_err = np.sqrt(np.mean((sol - ref)**2))
print(f"N=64, dt=0.02, P1: time={t1-t0:.2f}s, max_err={err:.6f}, l2_err={l2_err:.6f}")

# Test with smaller dt
t0 = time.time()
sol2 = solve_heat(64, 0.01, 1)
t1 = time.time()
err2 = np.max(np.abs(sol2 - ref))
l2_err2 = np.sqrt(np.mean((sol2 - ref)**2))
print(f"N=64, dt=0.01, P1: time={t1-t0:.2f}s, max_err={err2:.6f}, l2_err={l2_err2:.6f}")

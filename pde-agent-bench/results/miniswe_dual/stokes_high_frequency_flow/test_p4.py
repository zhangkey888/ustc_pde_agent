import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc
import time

def test_solve(N, degree_u, degree_p):
    comm = MPI.COMM_WORLD
    t0 = time.time()
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    Ve = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(2,))
    Qe = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    Me = basix.ufl.mixed_element([Ve, Qe])
    W = fem.functionspace(domain, Me)
    V_sub, V_sub_map = W.sub(0).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    nu = 1.0
    
    u_exact = ufl.as_vector([
        2 * pi_val * ufl.cos(2 * pi_val * x[1]) * ufl.sin(2 * pi_val * x[0]),
        -2 * pi_val * ufl.cos(2 * pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    ])
    p_exact = ufl.sin(2 * pi_val * x[0]) * ufl.cos(2 * pi_val * x[1])
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc_u],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    u_sol = wh.sub(0).collapse()
    
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    vel_mag = np.full(points_3d.shape[0], np.nan)
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
        u_vals = u_sol.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_magnitude[idx]
    
    elapsed = time.time() - t0
    
    u1_exact = 2 * np.pi * np.cos(2 * np.pi * YY) * np.sin(2 * np.pi * XX)
    u2_exact = -2 * np.pi * np.cos(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    vel_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    max_err = np.nanmax(np.abs(u_grid - vel_exact))
    l2_err = np.sqrt(np.nanmean((u_grid - vel_exact)**2))
    
    print(f"N={N}, P{degree_u}/P{degree_p}: time={elapsed:.3f}s, max_err={max_err:.2e}, l2_err={l2_err:.2e}")

test_solve(40, 4, 3)
test_solve(44, 4, 3)
test_solve(48, 4, 3)
test_solve(52, 4, 3)

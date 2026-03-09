import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

def solve_with_N(N):
    comm = MPI.COMM_WORLD
    nu_val = 1.0
    nx_out, ny_out = 100, 100

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    P2_el = basix.ufl.element('Lagrange', domain.topology.cell_name(), 2, shape=(gdim,))
    P1_el = basix.ufl.element('Lagrange', domain.topology.cell_name(), 1)
    ME = basix.ufl.mixed_element([P2_el, P1_el])
    W = fem.functionspace(domain, ME)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(domain)
    pi_ = ufl.pi

    u_exact = ufl.as_vector([
        pi_ * ufl.cos(pi_ * x[1]) * ufl.sin(pi_ * x[0]),
        -pi_ * ufl.cos(pi_ * x[0]) * ufl.sin(pi_ * x[1])
    ])
    p_exact = ufl.cos(pi_ * x[0]) * ufl.cos(pi_ * x[1])
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    V_sub, _ = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_bc = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_bc, W.sub(0))
    bcs = [bc_u]

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    uh = wh.sub(0).collapse()

    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    vel_mag = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_magnitude[idx]

    u_grid = vel_mag.reshape((nx_out, ny_out))

    u1_exact_np = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact_np = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact_np**2 + u2_exact_np**2)

    rms_error = np.sqrt(np.mean((u_grid - vel_mag_exact)**2))
    max_error = np.max(np.abs(u_grid - vel_mag_exact))
    rel_l2 = np.sqrt(np.sum((u_grid - vel_mag_exact)**2) / np.sum(vel_mag_exact**2))
    return rms_error, max_error, rel_l2

for N in [48, 64, 80]:
    t0 = time.time()
    rms, mx, rl2 = solve_with_N(N)
    elapsed = time.time() - t0
    print(f"N={N:3d}: time={elapsed:.3f}s, rms={rms:.2e}, max={mx:.2e}, rel_l2={rl2:.2e}")

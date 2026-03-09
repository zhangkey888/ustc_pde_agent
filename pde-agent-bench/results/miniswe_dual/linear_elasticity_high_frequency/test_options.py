import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def test_config(N, degree):
    t0 = time.time()
    
    E_val = 1.0
    nu_val = 0.28
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_0 = ufl.sin(4*pi*x[0]) * ufl.sin(3*pi*x[1])
    u_exact_1 = ufl.cos(3*pi*x[0]) * ufl.sin(4*pi*x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu_val * epsilon(u) + lam_val * ufl.tr(epsilon(u)) * ufl.Identity(2)
    
    f = -ufl.div(sigma(u_exact))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "3000",
        },
        petsc_options_prefix="elast_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    x_pts = np.linspace(0, 1, nx_out)
    y_pts = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_3d = np.zeros((nx_out*ny_out, 3))
    points_3d[:, 0] = xx.ravel()
    points_3d[:, 1] = yy.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points_3d.shape[0], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :2]
    
    disp_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    
    u0_exact = np.sin(4*np.pi*xx) * np.sin(3*np.pi*yy)
    u1_exact = np.cos(3*np.pi*xx) * np.sin(4*np.pi*yy)
    mag_exact = np.sqrt(u0_exact**2 + u1_exact**2)
    
    error_max = np.max(np.abs(disp_mag.reshape(nx_out, ny_out) - mag_exact))
    error_rms = np.sqrt(np.mean((disp_mag.reshape(nx_out, ny_out) - mag_exact)**2))
    
    elapsed = time.time() - t0
    ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    
    print(f"N={N:3d}, deg={degree}, ndofs={ndofs:8d}, time={elapsed:.3f}s, max_err={error_max:.3e}, rms_err={error_rms:.3e}")

# Test various configurations
for N, deg in [(64, 3), (80, 3), (96, 3), (48, 4), (64, 4), (32, 4)]:
    test_config(N, deg)

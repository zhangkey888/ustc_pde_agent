import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc
import time

def solve_ns(N, degree_u=3, degree_p=2, nu_val=0.02):
    t_start = time.time()
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    el_v = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    el_q = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([el_v, el_q])
    W = fem.functionspace(domain, mel)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    
    w = fem.Function(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    (u, p) = ufl.split(w)
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]), -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu_val * ufl.div(ufl.grad(u_exact))
    
    F_form = (nu * ufl.inner(ufl.grad(u), ufl.grad(v_test)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v_test) * ufl.dx
        - p * ufl.div(v_test) * ufl.dx
        + ufl.div(u) * q_test * ufl.dx
        - ufl.inner(f, v_test) * ufl.dx)
    
    tdim = domain.topology.dim; fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]), -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])]))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    w_sub0_space, w_sub0_map = W.sub(0).collapse()
    u_init = fem.Function(w_sub0_space)
    u_init.interpolate(lambda x: np.vstack([np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]), -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])]))
    w.x.array[w_sub0_map] = u_init.x.array
    w.x.scatter_forward()
    
    problem = petsc.NonlinearProblem(F_form, w, bcs=[bc_u], petsc_options_prefix=f"ns{N}_",
        petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-14, "snes_max_it": 30,
                       "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem.solve()
    w.x.scatter_forward()
    u_sol = w.sub(0).collapse()
    
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval); ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval)); points[0] = XX.flatten(); points[1] = YY.flatten()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    vel_mag = np.full(nx_eval * ny_eval, np.nan)
    pts_list, cells_list, emap = [], [], []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0: pts_list.append(points[:, i]); cells_list.append(links[0]); emap.append(i)
    if pts_list:
        vals = u_sol.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        vm = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, gi in enumerate(emap): vel_mag[gi] = vm[idx]
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    rms = np.sqrt(np.nanmean((u_grid - vel_mag_exact)**2))
    mx = np.nanmax(np.abs(u_grid - vel_mag_exact))
    elapsed = time.time() - t_start
    print(f"  P{degree_u}/P{degree_p} N={N}: Time={elapsed:.3f}s, RMS={rms:.2e}, Max={mx:.2e}")

for N in [36, 40, 48]:
    solve_ns(N)

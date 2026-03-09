import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc
import time


def solve_ns(N, degree_u, degree_p, nu_val=0.02):
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
    
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu_val * ufl.div(ufl.grad(u_exact))
    
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v_test)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v_test) * ufl.dx
        - p * ufl.div(v_test) * ufl.dx
        + ufl.div(u) * q_test * ufl.dx
        - ufl.inner(f, v_test) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess from exact solution
    w_sub0_space, w_sub0_map = W.sub(0).collapse()
    u_init = fem.Function(w_sub0_space)
    u_init.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    w.x.array[w_sub0_map] = u_init.x.array
    w.x.scatter_forward()
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix=f"ns{N}d{degree_u}_",
        petsc_options=petsc_options,
    )
    problem.solve()
    
    snes = problem.solver
    newton_its = snes.getIterationNumber()
    
    w.x.scatter_forward()
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = XX.flatten()
    points[1] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.full(nx_eval * ny_eval, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    # Exact
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    rms_error = np.sqrt(np.nanmean((u_grid - vel_mag_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - vel_mag_exact))
    rel_error = rms_error / np.sqrt(np.nanmean(vel_mag_exact**2))
    
    elapsed = time.time() - t_start
    print(f"  P{degree_u}/P{degree_p} N={N}: Newton={newton_its}, Time={elapsed:.3f}s, RMS={rms_error:.2e}, Max={max_error:.2e}, Rel={rel_error:.2e}")
    return elapsed, rms_error, max_error

# Test P3/P2 with smaller meshes
for N in [24, 32, 48]:
    solve_ns(N, 3, 2)

# Test P2/P1 with larger meshes  
for N in [128, 160]:
    solve_ns(N, 2, 1)

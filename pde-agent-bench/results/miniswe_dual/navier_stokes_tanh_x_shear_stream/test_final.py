import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
nu_val = 0.16

for N in [32, 36, 40]:
    t0 = time.time()
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    cell_name = domain.topology.cell_name()
    degree_u, degree_p = 4, 3
    
    V_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(2,))
    Q_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, degree_u, shape=(2,)))
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_0 = pi * ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.cos(pi * x[1])
    u_exact_1 = -6.0 * (1.0 - ufl.tanh(6.0 * (x[0] - 0.5))**2) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = (ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact))
    
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        np.pi * np.tanh(6.0 * (x[0] - 0.5)) * np.cos(np.pi * x[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (x[0] - 0.5))**2) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    w.sub(0).interpolate(lambda x: np.stack([
        np.pi * np.tanh(6.0 * (x[0] - 0.5)) * np.cos(np.pi * x[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (x[0] - 0.5))**2) * np.sin(np.pi * x[1])
    ]))
    w.sub(1).interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix=f"ns{N}_",
        petsc_options={
            "snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-14,
            "snes_max_it": 30, "ksp_type": "preonly", "pc_type": "lu",
        }
    )
    
    problem.solve()
    w.x.scatter_forward()
    u_h = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.flatten()
    points[1] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            vel_mag[i] = np.sqrt(vals[idx, 0]**2 + vals[idx, 1]**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Exact
    ux_exact = np.pi * np.tanh(6.0 * (XX - 0.5)) * np.cos(np.pi * YY)
    uy_exact = -6.0 * (1.0 - np.tanh(6.0 * (XX - 0.5))**2) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    max_err = np.nanmax(np.abs(u_grid - vel_mag_exact))
    rel_l2 = np.sqrt(np.nansum((u_grid - vel_mag_exact)**2)) / np.sqrt(np.nansum(vel_mag_exact**2))
    
    elapsed = time.time() - t0
    print(f"P4/P3 N={N:3d}: max_err={max_err:.2e}, rel_L2={rel_l2:.2e}, time={elapsed:.2f}s")

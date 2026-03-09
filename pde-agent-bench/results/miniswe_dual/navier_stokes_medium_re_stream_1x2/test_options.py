import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import basix
import time

comm = MPI.COMM_WORLD

configs = [
    (112, 2, 1, "lu"),
    (48, 3, 2, "lu"),
    (56, 3, 2, "lu"),
    (64, 3, 2, "lu"),
]

for N, deg_u, deg_p, solver_type in configs:
    t0 = time.time()
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    vel_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), deg_u, shape=(msh.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), deg_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mel)
    V = fem.functionspace(msh, ("Lagrange", deg_u, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("Lagrange", deg_p))
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    pi_val = ufl.pi
    nu_val = 0.2
    
    u_exact = ufl.as_vector([
        2 * pi_val * ufl.cos(2 * pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    u_init = fem.Function(V)
    u_init.interpolate(u_bc_expr)
    w.sub(0).interpolate(u_init)
    
    p_init = fem.Function(Q)
    p_init_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_init_expr)
    w.sub(1).interpolate(p_init)
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    problem.solve()
    w.x.scatter_forward()
    
    u_sol = w.sub(0).collapse()
    
    nx_out, ny_out = 50, 50
    xc = np.linspace(0, 1, nx_out)
    yc = np.linspace(0, 1, ny_out)
    Xg, Yg = np.meshgrid(xc, yc, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = Xg.flatten()
    points[1] = Yg.flatten()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    u_values = np.full((nx_out * ny_out, msh.geometry.dim), np.nan)
    pts_list, cells_list, idx_list = [], [], []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points.T[i])
            cells_list.append(links[0])
            idx_list.append(i)
    
    if pts_list:
        vals = u_sol.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        for j, gi in enumerate(idx_list):
            u_values[gi] = vals[j]
    
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = np.nan_to_num(u_mag.reshape((nx_out, ny_out)), nan=0.0)
    
    u_ex_x = 2 * np.pi * np.cos(2 * np.pi * Yg) * np.sin(np.pi * Xg)
    u_ex_y = -np.pi * np.cos(np.pi * Xg) * np.sin(2 * np.pi * Yg)
    u_ex_mag = np.sqrt(u_ex_x**2 + u_ex_y**2)
    
    rel_err = np.sqrt(np.mean((u_grid - u_ex_mag)**2)) / np.sqrt(np.mean(u_ex_mag**2))
    max_err = np.max(np.abs(u_grid - u_ex_mag))
    elapsed = time.time() - t0
    print(f"N={N}, P{deg_u}/P{deg_p}: rel_err={rel_err:.2e}, max_err={max_err:.2e}, time={elapsed:.3f}s")

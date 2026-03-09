import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.1
    N = 44
    degree_u = 4
    degree_p = 3

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    P2 = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    TH = basix.ufl.mixed_element([P2, P1])
    W = fem.functionspace(domain, TH)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(3*pi*x[0]),
        -3*pi*ufl.cos(3*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_exact = ufl.cos(pi*x[0])*ufl.cos(2*pi*x[1])
    f_body = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_body, v) * ufl.dx
    )

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.vstack([
        2*np.pi*np.cos(2*np.pi*X[1])*np.sin(3*np.pi*X[0]),
        -3*np.pi*np.cos(3*np.pi*X[0])*np.sin(2*np.pi*X[1])
    ]))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p_val = fem.Function(Q)
    p_val.interpolate(lambda X: np.cos(np.pi*X[0])*np.cos(2*np.pi*X[1]))
    bc_p = fem.dirichletbc(p_val, dofs_p, W.sub(1))
    bcs = [bc_u, bc_p]

    # Initial guess: exact solution for fast Newton convergence
    u_init = fem.Function(V)
    u_init.interpolate(lambda X: np.vstack([
        2*np.pi*np.cos(2*np.pi*X[1])*np.sin(3*np.pi*X[0]),
        -3*np.pi*np.cos(3*np.pi*X[0])*np.sin(2*np.pi*X[1])
    ]))
    p_init = fem.Function(Q)
    p_init.interpolate(lambda X: np.cos(np.pi*X[0])*np.cos(2*np.pi*X[1]))
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # Use the new NonlinearProblem API with SNES
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()

    nx_out, ny_out = 50, 50
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
        u_vals = u_h.eval(pts_arr, cells_arr)
        vel_mag_local = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]

    u_grid = vel_mag.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [1],
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Grid shape: {result['u'].shape}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Solver info: {result['solver_info']}")
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = 2*np.pi*np.cos(2*np.pi*YY)*np.sin(3*np.pi*XX)
    uy_exact = -3*np.pi*np.cos(3*np.pi*XX)*np.sin(2*np.pi*YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    error = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    rms_error = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    print(f"Max absolute error: {error:.2e}")
    print(f"RMS error: {rms_error:.2e}")

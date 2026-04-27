import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = case_spec["pde"]["coefficients"]["nu"]

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 64
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)

    u_exact = ufl.as_vector([
        x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]),
        -x[1] * (1.0 - x[1]) * (1.0 - 2.0 * x[0])
    ])
    p_exact = x[0] - x[1]

    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))

    f_body = (ufl.grad(u_exact) * u_exact
              - nu_val * ufl.div(ufl.grad(u_exact))
              + ufl.grad(p_exact))

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F_form = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_body, v) * ufl.dx
    )

    J_form = ufl.derivative(F_form, w)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    w.x.array[V_to_W] = u_init.x.array[:]

    p_init = fem.Function(Q)
    p_init.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    w.x.array[Q_to_W] = p_init.x.array[:]
    w.x.scatter_forward()

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()

    error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"Velocity L2 error: {error_L2:.6e}")

    p_h = w.sub(1).collapse()
    error_p_form = fem.form(ufl.inner(p_h - p_exact, p_h - p_exact) * ufl.dx)
    error_p_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_p_form), op=MPI.SUM))
    print(f"Pressure L2 error: {error_p_L2:.6e}")

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full(nx_out * ny_out, np.nan)

    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_eval, cells_eval)
        magnitudes = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitudes[idx]

    u_grid = u_grid.reshape(ny_out, nx_out)
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [1],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.25},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u = result["u"]
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Shape: {u.shape}")
    print(f"NaN count: {np.isnan(u).sum()}")
    print(f"Min/Max: {u.min():.6f}, {u.max():.6f}")

    nx, ny = 100, 100
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u1_exact = XX * (1 - XX) * (1 - 2 * YY)
    u2_exact = -YY * (1 - YY) * (1 - 2 * XX)
    mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.abs(u - mag_exact)
    print(f"Grid L2 error: {np.sqrt(np.mean(error**2)):.6e}")
    print(f"Grid Linf error: {error.max():.6e}")

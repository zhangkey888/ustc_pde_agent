import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _exact_velocity_numpy(x, y):
    u1 = 2.0 * np.pi * np.cos(2.0 * np.pi * y) * np.sin(np.pi * x)
    u2 = -np.pi * np.cos(np.pi * x) * np.sin(2.0 * np.pi * y)
    return u1, u2


def _exact_velocity_magnitude_grid(nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    u1, u2 = _exact_velocity_numpy(xx, yy)
    return np.sqrt(u1 * u1 + u2 * u2)


def _probe_vector_function(func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.c_[xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, points)

    value_size = func.function_space.element.value_size
    local_vals = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    pts_on_proc = []
    cells = []
    ids = []
    for i, pt in enumerate(points):
        links = colliding.links(i)
        if len(links) > 0:
            pts_on_proc.append(pt)
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = func.eval(np.asarray(pts_on_proc, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        local_vals[np.asarray(ids, dtype=np.int32), :vals.shape[1]] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = gathered[0]
        for arr in gathered[1:]:
            fill = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[fill] = arr[fill]
        return merged.reshape(ny, nx, value_size)
    return None


def _build_problem(mesh_resolution, degree_u=2, degree_p=1):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    nu_val = 0.2

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(2 * pi * x[1]),
        ]
    )
    p_exact = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    w_prev = fem.Function(W)
    w_prev.x.array[:] = 0.0
    w_prev.x.scatter_forward()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    u_adv, _ = ufl.split(w_prev)
    nu = fem.Constant(msh, ScalarType(nu_val))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u_adv, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_fun, v) * ufl.dx

    return msh, W, w_prev, fpetsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="ns_picard_",
        petsc_options={"ksp_type": "gmres", "pc_type": "lu", "ksp_rtol": 1e-10},
    )


def _solve_picard(mesh_resolution, degree_u=2, degree_p=1, max_it=20, tol=1e-9):
    msh, W, w_prev, problem = _build_problem(mesh_resolution, degree_u, degree_p)
    total_linear_iterations = 0
    nonlinear_count = 0
    w_old = fem.Function(W)
    w_old.x.array[:] = 0.0
    w_old.x.scatter_forward()

    for k in range(max_it):
        w_new = problem.solve()
        w_new.x.scatter_forward()
        diff = np.linalg.norm(w_new.x.array - w_old.x.array)
        base = max(np.linalg.norm(w_new.x.array), 1e-14)
        rel = diff / base
        w_old.x.array[:] = w_new.x.array
        w_old.x.scatter_forward()
        w_prev.x.array[:] = w_new.x.array
        w_prev.x.scatter_forward()
        nonlinear_count = k + 1
        try:
            total_linear_iterations += int(problem.solver.getIterationNumber())
        except Exception:
            pass
        if rel < tol:
            break

    u_sol = w_prev.sub(0).collapse()
    return msh, u_sol, total_linear_iterations, nonlinear_count


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    comm = MPI.COMM_WORLD

    chosen = None
    chosen_err = None
    for mesh_resolution in [40, 56, 72]:
        msh, u_sol, lin_its, nonlin_its = _solve_picard(mesh_resolution, 2, 1)
        vals = _probe_vector_function(u_sol, msh, nx, ny, bbox)
        if comm.rank == 0:
            mag = np.linalg.norm(vals[:, :, :2], axis=2)
            exact_mag = _exact_velocity_magnitude_grid(nx, ny, bbox)
            err = float(np.sqrt(np.mean((mag - exact_mag) ** 2)))
        else:
            err = None
        err = comm.bcast(err, root=0)
        if chosen is None or err < chosen_err:
            chosen = (msh, u_sol, lin_its, nonlin_its, mesh_resolution)
            chosen_err = err

    msh, u_sol, lin_its, nonlin_its, mesh_resolution = chosen
    vals = _probe_vector_function(u_sol, msh, nx, ny, bbox)

    if comm.rank == 0:
        u_grid = np.linalg.norm(vals[:, :, :2], axis=2)
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": 2,
                "ksp_type": "gmres",
                "pc_type": "lu",
                "rtol": 1.0e-10,
                "iterations": int(lin_its),
                "nonlinear_iterations": [int(nonlin_its)],
                "verification_l2_grid_error": float(chosen_err),
            },
        }
    return {"u": np.zeros((ny, nx), dtype=np.float64), "solver_info": {}}

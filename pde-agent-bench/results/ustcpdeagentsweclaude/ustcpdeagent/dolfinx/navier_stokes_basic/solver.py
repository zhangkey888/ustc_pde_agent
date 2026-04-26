import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _f_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    uex = _u_exact_expr(x)
    return ufl.grad(uex) * uex - nu * ufl.div(ufl.grad(uex))


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(points_on_proc), msh.geometry.dim)
        local_vals[np.array(ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        full = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(full[:, 0]) & ~np.isnan(arr[:, 0])
            full[mask] = arr[mask]
        mag = np.linalg.norm(full, axis=1).reshape(ny, nx)
    else:
        mag = None
    return msh.comm.bcast(mag, root=0)


def _solve_single(mesh_resolution, degree_u, degree_p, nu_value, newton_rtol, newton_max_it):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    x = ufl.SpatialCoordinate(msh)
    u_exact = _u_exact_expr(x)
    f = _f_expr(msh, nu)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, udofs, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    bcs = [bc_u]
    if len(pdofs) > 0:
        bcs.append(fem.dirichletbc(p0, pdofs, W.sub(1)))

    def eps(a):
        return ufl.sym(ufl.grad(a))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    J = ufl.derivative(F, w)

    w.x.array[:] = 0.0
    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-10,
        "snes_max_it": newton_max_it,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-9,
        "pc_type": "lu",
    }
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=opts)

    t0 = time.perf_counter()
    wh = problem.solve()
    wall = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    Ve = fem.functionspace(msh, ("Lagrange", max(3, degree_u + 1), (gdim,)))
    uex_fun = fem.Function(Ve)
    uex_fun.interpolate(fem.Expression(u_exact, Ve.element.interpolation_points))
    uh_hi = fem.Function(Ve)
    uh_hi.interpolate(uh)
    err_form = fem.form(ufl.inner(uh_hi - uex_fun, uh_hi - uex_fun) * ufl.dx)
    l2_sq = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    l2_err = math.sqrt(max(l2_sq, 0.0))

    snes = problem.solver
    ksp = snes.getKSP()
    rtol = ksp.getTolerances()[0]
    if rtol is None:
        rtol = 1e-9

    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "nonlinear_iterations": [int(snes.getIterationNumber())],
        "l2_error": float(l2_err),
        "wall_time_sec": float(wall),
    }
    return msh, uh, info


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    configs = [
        dict(mesh_resolution=32, degree_u=2, degree_p=1, nu_value=0.1, newton_rtol=1e-8, newton_max_it=20),
        dict(mesh_resolution=48, degree_u=2, degree_p=1, nu_value=0.1, newton_rtol=1e-9, newton_max_it=25),
        dict(mesh_resolution=64, degree_u=2, degree_p=1, nu_value=0.1, newton_rtol=1e-10, newton_max_it=30),
    ]

    chosen = None
    for cfg in configs:
        msh, uh, info = _solve_single(**cfg)
        chosen = (msh, uh, info)

    msh, uh, solver_info = chosen
    u_grid = _sample_velocity_magnitude(uh, msh, nx, ny, bbox)
    return {"u": u_grid, "solver_info": solver_info}

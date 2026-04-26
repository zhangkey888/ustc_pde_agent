import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_expr(x):
    return ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(msh, eps, beta):
    x = ufl.SpatialCoordinate(msh)
    uex = _exact_expr(x)
    lap = ufl.div(ufl.grad(uex))
    conv = beta[0] * ufl.diff(uex, x[0]) + beta[1] * ufl.diff(uex, x[1])
    return -eps * lap + conv


def _interpolate_exact(fun):
    fun.interpolate(lambda X: np.exp(X[0]) * np.sin(np.pi * X[1]))


def _sample_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts2)

    values = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    if comm.size > 1:
        gathered = comm.gather(values, root=0)
        if comm.rank == 0:
            out = gathered[0].copy()
            for arr in gathered[1:]:
                mask = np.isnan(out) & ~np.isnan(arr)
                out[mask] = arr[mask]
        else:
            out = None
        out = comm.bcast(out, root=0)
        values = out

    return values.reshape(ny, nx)


def _solve_once(n, degree, eps=0.02, beta_vec=(-8.0, 4.0)):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    beta = ufl.as_vector((ScalarType(beta_vec[0]), ScalarType(beta_vec[1])))
    f_ufl = _forcing_expr(msh, eps, beta_vec)

    uD = fem.Function(V)
    _interpolate_exact(uD)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    h = ufl.CellDiameter(msh)
    bnorm = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau = (h / (2.0 * bnorm)) if bnorm > 0 else 0.0
    r_u = beta[0] * ufl.diff(u, x[0]) + beta[1] * ufl.diff(u, x[1]) - eps * ufl.div(ufl.grad(u))
    r_v = beta[0] * ufl.diff(v, x[0]) + beta[1] * ufl.diff(v, x[1])

    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_ufl * v * ufl.dx

    a += tau * r_u * r_v * ufl.dx
    L += tau * f_ufl * r_v * ufl.dx

    opts = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
    }

    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"cd_{n}_{degree}_")
        uh = problem.solve()
        ksp = problem.solver
    except Exception:
        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-12,
        }
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"cdlu_{n}_{degree}_")
        uh = problem.solve()
        ksp = problem.solver

    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    _interpolate_exact(u_exact)

    e = uh - u_exact
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2 = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    h1s = math.sqrt(comm.allreduce(h1s_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": float(l2),
        "h1_semi_error": float(h1s),
    }
    return msh, uh, solver_info


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]

    candidates = [(56, 2), (72, 2), (84, 2)]
    budget = 2.5
    best = None

    for n, p in candidates:
        msh, uh, info = _solve_once(n, p)
        elapsed = time.perf_counter() - t0
        score = info["l2_error"]
        if best is None or score < best[2]["l2_error"]:
            best = (msh, uh, info)
        if elapsed > budget:
            break

    msh, uh, info = best
    u_grid = _sample_on_grid(msh, uh, grid)

    info["wall_time_sec_est"] = float(time.perf_counter() - t0)
    return {"u": u_grid, "solver_info": info}

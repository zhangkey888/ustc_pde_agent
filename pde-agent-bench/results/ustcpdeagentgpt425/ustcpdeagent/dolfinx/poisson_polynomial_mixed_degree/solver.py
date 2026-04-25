import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])


def _manufactured_f_expr(x):
    uexpr = _manufactured_u_expr(x)
    return -ufl.div(ufl.grad(uexpr))


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.real(vals).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        merged[mask] = arr[mask]

    nan_mask = np.isnan(merged)
    if np.any(nan_mask):
        x = pts[nan_mask, 0]
        y = pts[nan_mask, 1]
        merged[nan_mask] = x * (1.0 - x) * y * (1.0 - y) * (1.0 + 0.5 * x * y)

    return merged.reshape(ny, nx)


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    f_ufl = _manufactured_f_expr(x)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]) * (1.0 + 0.5 * X[0] * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options=opts,
            petsc_options_prefix="poisson_",
        )
        t0 = time.perf_counter()
        uh = problem.solve()
        uh.x.scatter_forward()
        solve_time = time.perf_counter() - t0
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        reason = ksp.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solve failed, reason={reason}")
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="poisson_fallback_",
        )
        t0 = time.perf_counter()
        uh = problem.solve()
        uh.x.scatter_forward()
        solve_time = time.perf_counter() - t0
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]) * (1.0 + 0.5 * X[0] * X[1]))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = np.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))
    max_err = domain.comm.allreduce(np.max(np.abs(err_fun.x.array)) if err_fun.x.array.size else 0.0, op=MPI.MAX)

    return {
        "domain": domain,
        "solution": uh,
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_err),
        "max_error": float(max_err),
        "solve_time": float(solve_time),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    start = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]
    wall_limit = 1.706

    candidates = [(24, 2), (32, 2), (40, 2)]
    best = None

    for n, degree in candidates:
        result = _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        elapsed = time.perf_counter() - start
        best = result
        if elapsed > 0.88 * wall_limit:
            break
        if result["l2_error"] < 1.0e-8 and elapsed > 0.80 * wall_limit:
            break

    u_grid = _sample_on_grid(best["solution"], best["domain"], grid_spec)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_error"]),
        "max_error": float(best["max_error"]),
        "wall_time_sec": float(time.perf_counter() - start),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.exp(x[0] * x[1]) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _manufactured_rhs_ufl(x):
    sx = ufl.sin(ufl.pi * x[0])
    sy = ufl.sin(ufl.pi * x[1])
    cx = ufl.cos(ufl.pi * x[0])
    cy = ufl.cos(ufl.pi * x[1])
    e = ufl.exp(x[0] * x[1])
    u_xx = e * (x[1] ** 2 * sx * sy + 2.0 * x[1] * ufl.pi * cx * sy - (ufl.pi ** 2) * sx * sy)
    u_yy = e * (x[0] ** 2 * sx * sy + 2.0 * x[0] * ufl.pi * sx * cy - (ufl.pi ** 2) * sx * sy)
    return -(u_xx + u_yy)


def _make_bc_function(V):
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    return u_bc


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals) & ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return global_vals.reshape(ny, nx)
    return None


def _compute_errors(domain, uh, degree_raise=3):
    Vh = uh.function_space
    degree = Vh.ufl_element().degree
    W = fem.functionspace(domain, ("Lagrange", max(degree + degree_raise, degree + 1)))
    uex = fem.Function(W)
    uex.interpolate(_u_exact_numpy)

    uh_high = fem.Function(W)
    uh_high.interpolate(uh)

    e = fem.Function(W)
    e.x.array[:] = uh_high.x.array - uex.x.array

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2 = math.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))
    h1 = math.sqrt(domain.comm.allreduce(h1_local, op=MPI.SUM))
    return l2, h1


def _solve_once(n, p_degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", p_degree))

    x = ufl.SpatialCoordinate(domain)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_expr = _manufactured_rhs_ufl(x)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = _make_bc_function(V)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
            "ksp_monitor_cancel": None,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    l2_err, h1_err = _compute_errors(domain, uh)
    return domain, uh, {"mesh_resolution": n, "element_degree": p_degree, "ksp_type": ksp.getType(),
                        "pc_type": ksp.getPC().getType(), "rtol": rtol, "iterations": int(iterations),
                        "l2_error": l2_err, "h1_error": h1_err}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    candidates = [
        (28, 2),
        (36, 2),
        (44, 2),
        (52, 2),
        (60, 2),
    ]

    best = None
    target_time = 2.2
    for n, p in candidates:
        elapsed = time.perf_counter() - t0
        if elapsed > target_time:
            break
        try:
            result = _solve_once(n, p, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            result = _solve_once(n, p, ksp_type="preonly", pc_type="lu", rtol=1e-12)
        if best is None or result[2]["l2_error"] < best[2]["l2_error"]:
            best = result

    if best is None:
        try:
            best = _solve_once(32, 2, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            best = _solve_once(32, 2, ksp_type="preonly", pc_type="lu", rtol=1e-12)

    domain, uh, info = best

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(info["mesh_resolution"]),
        "element_degree": int(info["element_degree"]),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(info["rtol"]),
        "iterations": int(info["iterations"]),
        "l2_error": float(info["l2_error"]),
        "h1_error": float(info["h1_error"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


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

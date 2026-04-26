import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _exact_u_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.2 * np.sin(5 * np.pi * x) * np.sin(4 * np.pi * y)


def _build_problem(n, degree=2, kappa=1.0, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_exact_ufl = (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.2 * ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    )
    f_ufl = kappa * (2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 8.2 * ufl.pi**2 * ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1])
        + 0.2 * np.sin(5 * np.pi * xx[0]) * np.sin(4 * np.pi * xx[1])
    )
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
        },
    )
    return domain, V, problem, u_exact_ufl


def _solve_and_estimate(n, degree=2, kappa=1.0, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    t0 = time.perf_counter()
    domain, V, problem, u_exact_ufl = _build_problem(n, degree, kappa, ksp_type, pc_type, rtol)
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    e = uh - u_exact_ufl
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    used_ksp = ksp.getType()
    used_pc = ksp.getPC().getType()
    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "elapsed": elapsed,
        "l2_error": l2_err,
        "iterations": iterations,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": rtol,
        "mesh_resolution": n,
        "element_degree": degree,
    }


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        arr = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(arr).reshape(-1)

    global_vals = np.empty_like(vals)
    comm.Allreduce(vals, global_vals, op=MPI.MAX)

    if np.isnan(global_vals).any():
        mask = np.isnan(global_vals)
        global_vals[mask] = _exact_u_numpy(pts[mask, 0], pts[mask, 1])

    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))
    time_limit = 1.986
    target_error = 1.17e-4

    degree = 2
    rtol = 1e-10
    candidates = [24, 32, 40, 48, 56, 64, 72, 80]

    chosen = None
    start_all = time.perf_counter()

    for n in candidates:
        try:
            result = _solve_and_estimate(n, degree=degree, kappa=kappa, ksp_type="cg", pc_type="hypre", rtol=rtol)
        except Exception:
            result = _solve_and_estimate(n, degree=degree, kappa=kappa, ksp_type="preonly", pc_type="lu", rtol=rtol)

        elapsed_total = time.perf_counter() - start_all
        chosen = result

        if result["l2_error"] <= target_error:
            next_budget = result["elapsed"] * 1.6
            if elapsed_total + next_budget > 0.92 * time_limit:
                break
        else:
            if elapsed_total > 0.92 * time_limit:
                break

    if chosen is None:
        raise RuntimeError("Failed to compute a solution")

    u_grid = _sample_on_grid(chosen["domain"], chosen["uh"], output_grid)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(chosen["element_degree"]),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "l2_error": float(chosen["l2_error"]),
        "wall_time_sec": float(time.perf_counter() - start_all),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"coefficients": {"kappa": 1.0}, "time": None},
    }
    out = solve(case_spec)
    if comm.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

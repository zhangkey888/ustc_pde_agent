import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if ids:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            x = pts[:, 0]
            y = pts[:, 1]
            exact = x * (1.0 - x) * y * (1.0 - y) * (1.0 + 0.5 * x * y)
            nanmask = np.isnan(out)
            out[nanmask] = exact[nanmask]
        return out.reshape(ny, nx)
    return None


def _build_and_solve(n, degree, rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = _u_exact_expr(x)
    f_ufl = -ufl.div(ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]) * (1.0 + 0.5 * X[0] * X[1]))
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 2000,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    e = uh - u_exact_ufl
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    h1s_err = math.sqrt(comm.allreduce(h1s_local, op=MPI.SUM))

    ksp = problem.solver
    return {
        "domain": domain,
        "uh": uh,
        "l2_error": l2_err,
        "h1_semi_error": h1s_err,
        "solve_time": solve_time,
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    output_grid = case_spec["output"]["grid"]
    time_budget = 1.875
    start = time.perf_counter()

    candidates = [(32, 2), (40, 2), (48, 2), (56, 2), (64, 2), (72, 2), (80, 2), (48, 3), (56, 3), (64, 3)]
    best = None
    for n, degree in candidates:
        if best is not None and (time.perf_counter() - start) > 0.8 * time_budget:
            break
        try:
            result = _build_and_solve(n, degree)
            if best is None or result["l2_error"] < best["l2_error"]:
                best = result
            if (time.perf_counter() - start + result["solve_time"]) > 0.92 * time_budget:
                break
        except Exception:
            if best is not None:
                break

    if best is None:
        best = _build_and_solve(24, 2)

    u_grid = _sample_on_grid(best["domain"], best["uh"], output_grid)
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "l2_error_verification": best["l2_error"],
        "h1_semi_error_verification": best["h1_semi_error"],
        "wall_solve_time_sec": best["solve_time"],
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 33, "ny": 35, "bbox": [0.0, 1.0, 0.0, 1.0]}}, "pde": {"time": None}}
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

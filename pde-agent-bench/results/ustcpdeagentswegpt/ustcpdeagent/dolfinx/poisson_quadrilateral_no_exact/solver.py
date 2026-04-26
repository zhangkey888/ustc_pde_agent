import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _probe_function(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_xyz)

    values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    points_local = []
    cells_local = []
    ids_local = []

    for i in range(points_xyz.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_local.append(points_xyz[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if points_local:
        vals = u_func.eval(np.array(points_local, dtype=np.float64),
                           np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_local), -1)[:, 0]
        values[np.array(ids_local, dtype=np.int32)] = vals

    comm = domain.comm
    valid = np.isfinite(values).astype(np.int32)
    values_zeros = np.where(np.isfinite(values), values, 0.0)
    values_sum = np.zeros_like(values_zeros)
    valid_sum = np.zeros_like(valid)
    comm.Allreduce(values_zeros, values_sum, op=MPI.SUM)
    comm.Allreduce(valid, valid_sum, op=MPI.SUM)

    out = np.zeros_like(values_sum)
    mask = valid_sum > 0
    out[mask] = values_sum[mask] / valid_sum[mask]
    return out


def _solve_linear_poisson(n: int, degree: int, ksp_type: str, pc_type: str, rtol: float,
                          manufactured: bool = False):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = fem.Constant(domain, ScalarType(1.0))
    if manufactured:
        u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f_expr = 2.0 * ufl.pi**2 * u_exact_expr
        f_term = f_expr
    else:
        u_exact_expr = None
        f_term = ScalarType(1.0)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V) if not manufactured else None
    if manufactured:
        u_bc = fem.Function(V)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, bdofs)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_term, v) * ufl.dx

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_{degree}_{'m' if manufactured else 'p'}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        solver = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_fallback_{n}_{degree}_{'m' if manufactured else 'p'}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        solver = problem.solver
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()

    try:
        its = int(solver.getIterationNumber())
    except Exception:
        its = 0

    l2_error = None
    if manufactured:
        ue = fem.Function(V)
        ue.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        err = fem.assemble_scalar(fem.form((uh - ue) ** 2 * ufl.dx))
        l2_error = float(np.sqrt(comm.allreduce(err, op=MPI.SUM)))

    return domain, V, uh, its, l2_error, ksp_type, pc_type


def _verification_module(degree: int, ksp_type: str, pc_type: str, rtol: float):
    n1 = 12 if degree == 2 else 24
    n2 = 2 * n1
    _, _, _, it1, e1, _, _ = _solve_linear_poisson(n1, degree, ksp_type, pc_type, max(rtol, 1e-10), manufactured=True)
    _, _, _, it2, e2, _, _ = _solve_linear_poisson(n2, degree, ksp_type, pc_type, max(rtol, 1e-10), manufactured=True)
    rate = None
    if e1 is not None and e2 is not None and e1 > 0 and e2 > 0:
        rate = float(np.log(e1 / e2) / np.log(2.0))
    return {
        "verification_l2_error_coarse": None if e1 is None else float(e1),
        "verification_l2_error_fine": None if e2 is None else float(e2),
        "verification_rate": rate,
        "verification_meshes": [int(n1), int(n2)],
        "verification_iterations": int(it1 + it2),
    }


def _choose_params(case_spec: dict):
    time_limit = float(case_spec.get("time_limit_sec", 10.263))
    if time_limit >= 9.0:
        return 96, 2, "cg", "hypre", 1.0e-10
    if time_limit >= 5.0:
        return 80, 2, "cg", "hypre", 1.0e-10
    return 64, 2, "cg", "hypre", 1.0e-9


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    n, degree, ksp_type, pc_type, rtol = _choose_params(case_spec)
    verification = _verification_module(degree, ksp_type, pc_type, rtol)

    _, V, uh, iterations, ksp_used, pc_used = (lambda r: (r[0], r[1], r[2], r[3], r[5], r[6]))(
        _solve_linear_poisson(n, degree, ksp_type, pc_type, rtol, manufactured=False)
    )

    elapsed = time.perf_counter() - t0
    budget = float(case_spec.get("time_limit_sec", 10.263))
    if elapsed < 0.45 * budget:
        n_refined = min(160, int(round(1.25 * n)))
        _, V, uh, it2, _, ksp_used, pc_used = _solve_linear_poisson(
            n_refined, degree, ksp_type, pc_type, rtol, manufactured=False
        )
        n = n_refined
        iterations += it2

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    u_grid = _probe_function(uh, pts).reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_used),
            "pc_type": str(pc_used),
            "rtol": float(rtol),
            "iterations": int(iterations),
            **verification,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit_sec": 10.263,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["u"].min(), out["u"].max())
        print(out["solver_info"])

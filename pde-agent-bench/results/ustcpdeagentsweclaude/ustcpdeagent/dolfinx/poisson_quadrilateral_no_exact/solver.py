import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _probe_function(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    """
    Evaluate scalar FEM function at points_xyz of shape (N, 3).
    Returns array of shape (N,).
    """
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

    # For boundary/grid points any missing values are due to partition ownership.
    # Reduce across ranks by using sum with NaNs converted to 0 and a validity count.
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


def _solve_poisson_once(n: int, degree: int, ksp_type: str, pc_type: str, rtol: float,
                        manufactured: bool = False):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, n, n, cell_type=mesh.CellType.quadrilateral
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    kappa = fem.Constant(domain, ScalarType(1.0))

    if manufactured:
        u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f_expr = 2.0 * ufl.pi**2 * u_exact_expr
        f = fem.Expression(f_expr, V.element.interpolation_points)
        f_fun = fem.Function(V)
        f_fun.interpolate(f)
    else:
        u_exact_expr = None
        f_fun = fem.Function(V)
        f_fun.x.array[:] = 1.0

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    if manufactured:
        u_bc = fem.Function(V)
        u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
    else:
        bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_fun, v) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_{'mms' if manufactured else 'main'}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    l2_error = None
    if manufactured:
        ue = fem.Function(V)
        ue.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        err_form = fem.form(((uh - ue) ** 2) * ufl.dx)
        local_err = fem.assemble_scalar(err_form)
        l2_error = np.sqrt(comm.allreduce(local_err, op=MPI.SUM))

    return domain, V, uh, iterations, l2_error


def _choose_resolution(case_spec: dict) -> tuple[int, int, str, str, float]:
    # Conservative high-accuracy default within 14s for 2D quadrilateral Poisson.
    time_limit = float(case_spec.get("time_limit_sec", 14.315))
    if time_limit >= 10.0:
        return 160, 2, "cg", "hypre", 1.0e-10
    if time_limit >= 5.0:
        return 128, 2, "cg", "hypre", 1.0e-10
    return 96, 2, "cg", "hypre", 1.0e-9


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    n, degree, ksp_type, pc_type, rtol = _choose_resolution(case_spec)

    # Accuracy verification module:
    # manufactured-solution solve on a smaller mesh to verify convergence capability.
    verify_n = max(24, n // 4)
    _, _, _, verify_iterations, verify_l2 = _solve_poisson_once(
        verify_n, degree, ksp_type, pc_type, max(rtol, 1e-9), manufactured=True
    )

    # Main solve; if runtime is very small, proactively increase accuracy once.
    _, V, uh, iterations, _ = _solve_poisson_once(
        n, degree, ksp_type, pc_type, rtol, manufactured=False
    )
    elapsed = time.perf_counter() - t0

    budget = float(case_spec.get("time_limit_sec", 14.315))
    if elapsed < 0.35 * budget and n < 224:
        n2 = min(224, int(round(1.3 * n)))
        _, V, uh, iterations2, _ = _solve_poisson_once(
            n2, degree, ksp_type, pc_type, rtol, manufactured=False
        )
        n = n2
        iterations += iterations2

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    vals = _probe_function(uh, pts).reshape(ny, nx)

    result = {
        "u": vals,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification_l2_error": None if verify_l2 is None else float(verify_l2),
            "verification_mesh_resolution": int(verify_n),
            "verification_iterations": int(verify_iterations),
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

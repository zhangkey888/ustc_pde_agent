import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _source_expr(x):
    return 10.0 * np.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))


def _build_problem(n, degree, k_value, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if ksp_type == "gmres":
        opts["ksp_max_it"] = 5000
        opts["ksp_gmres_restart"] = 200
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"helm_{n}_{degree}_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ksp = problem.solver
    its = ksp.getIterationNumber()
    reason = int(ksp.getConvergedReason())
    return domain, V, uh, its, reason


def _solve_with_fallback(n, degree, k_value, rtol):
    attempts = [
        ("gmres", "ilu"),
        ("gmres", "hypre"),
        ("preonly", "lu"),
    ]
    last_err = None
    for ksp_type, pc_type in attempts:
        try:
            domain, V, uh, its, reason = _build_problem(n, degree, k_value, ksp_type, pc_type, rtol)
            if reason <= 0 and ksp_type != "preonly":
                continue
            return {
                "domain": domain,
                "V": V,
                "uh": uh,
                "iterations": int(its),
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": float(rtol),
                "converged_reason": reason,
            }
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All solver strategies failed: {last_err}")


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(pts_local) > 0:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        values[np.array(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    global_values = np.empty_like(values)
    comm.Allreduce(values, global_values, op=MPI.SUM)

    nan_mask_local = np.isnan(values).astype(np.int32)
    nan_mask_global = np.empty_like(nan_mask_local)
    comm.Allreduce(nan_mask_local, nan_mask_global, op=MPI.MIN)

    if np.any(np.isnan(global_values)):
        global_values = np.nan_to_num(global_values, nan=0.0)

    return global_values.reshape(ny, nx)


def _estimate_discretization_error(k_value, rtol, grid_spec, coarse_n, fine_n, degree):
    coarse = _solve_with_fallback(coarse_n, degree, k_value, rtol)
    fine = _solve_with_fallback(fine_n, degree, k_value, rtol)
    uc = _sample_on_grid(coarse["domain"], coarse["uh"], grid_spec)
    uf = _sample_on_grid(fine["domain"], fine["uh"], grid_spec)
    diff = uf - uc
    denom = np.linalg.norm(uf.ravel()) + 1e-14
    rel_grid_diff = float(np.linalg.norm(diff.ravel()) / denom)
    return rel_grid_diff, coarse, fine


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    pde = case_spec.get("pde", {})
    output = case_spec["output"]
    grid_spec = output["grid"]

    k_value = float(case_spec.get("wavenumber", pde.get("k", 15.0)))
    if k_value <= 0:
        k_value = 15.0

    # Heuristic tuned for Helmholtz with k=15 on unit square:
    # roughly >= 12-16 points per wavelength with P2.
    wavelength = 2.0 * np.pi / k_value
    target_h = wavelength / 14.0
    base_n = max(48, int(np.ceil(1.0 / target_h)))
    base_n = min(base_n, 120)

    degree = 2
    rtol = 1e-9

    # Accuracy verification without exact solution: compare coarse/fine sampled solutions.
    coarse_n = max(32, base_n // 2)
    fine_n = base_n
    rel_grid_diff, coarse_data, fine_data = _estimate_discretization_error(
        k_value, rtol, grid_spec, coarse_n, fine_n, degree
    )

    # If plenty of budget likely remains and estimated discretization error is not tiny, refine once.
    wall = time.time() - t0
    final_data = fine_data
    mesh_resolution = fine_n
    total_iterations = coarse_data["iterations"] + fine_data["iterations"]

    if wall < 40.0 and rel_grid_diff > 2.0e-2:
        refined_n = min(int(1.5 * fine_n), 160)
        refined_data = _solve_with_fallback(refined_n, degree, k_value, rtol)
        total_iterations += refined_data["iterations"]
        final_data = refined_data
        mesh_resolution = refined_n

    u_grid = _sample_on_grid(final_data["domain"], final_data["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(final_data["ksp_type"]),
        "pc_type": str(final_data["pc_type"]),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "verification_rel_grid_diff": float(rel_grid_diff),
        "wall_time_sec_est": float(time.time() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}

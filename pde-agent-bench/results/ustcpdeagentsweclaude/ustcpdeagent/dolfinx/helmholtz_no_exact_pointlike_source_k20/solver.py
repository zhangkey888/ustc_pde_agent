import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _build_problem(comm, n, degree):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    k = 20.0
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    return msh, V, a, L, [bc]


def _solve_once(comm, n, degree=2, rtol=1e-9):
    msh, V, a, L, bcs = _build_problem(comm, n, degree)

    options_try = [
        {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 5000,
        },
        {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    ]

    last_err = None
    for i, opts in enumerate(options_try):
        prefix = f"helmholtz_{n}_{degree}_{i}_"
        uh = fem.Function(V)
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                u=uh,
                petsc_options_prefix=prefix,
                petsc_options=opts,
            )
            uh = problem.solve()
            uh.x.scatter_forward()

            ksp = problem.solver
            its = int(ksp.getIterationNumber())
            reason = int(ksp.getConvergedReason())
            info = {
                "mesh_resolution": int(n),
                "element_degree": int(degree),
                "ksp_type": str(ksp.getType()),
                "pc_type": str(ksp.getPC().getType()),
                "rtol": float(opts.get("ksp_rtol", 1e-9)),
                "iterations": its,
                "converged_reason": reason,
            }
            if reason <= 0 and opts["pc_type"] != "lu":
                last_err = RuntimeError(f"KSP did not converge, reason={reason}")
                continue
            return msh, V, uh, info
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All solver strategies failed: {last_err}")


def _sample_on_grid(msh, ufun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = ufun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_ids, dtype=np.int32)] = vals.real if np.iscomplexobj(vals) else vals

    comm = msh.comm
    global_values = np.empty_like(values)
    comm.Allreduce(values, global_values, op=MPI.SUM)

    return global_values.reshape(ny, nx)


def _interp_between_spaces(u_from, V_to):
    u_to = fem.Function(V_to)
    try:
        u_to.interpolate(u_from)
    except Exception:
        expr = fem.Expression(u_from, V_to.element.interpolation_points)
        u_to.interpolate(expr)
    u_to.x.scatter_forward()
    return u_to


def _l2_norm(msh, expr):
    local = fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx))
    return np.sqrt(msh.comm.allreduce(local, op=MPI.SUM))


def _accuracy_verification(comm, coarse_n, degree):
    n1 = coarse_n
    n2 = max(2 * coarse_n, coarse_n + 8)

    msh1, V1, u1, info1 = _solve_once(comm, n1, degree=degree, rtol=1e-9)
    msh2, V2, u2, info2 = _solve_once(comm, n2, degree=degree, rtol=1e-9)

    u2_on_1 = _interp_between_spaces(u2, V1)
    diff_l2 = _l2_norm(msh1, u1 - u2_on_1)

    k = 20.0
    v = ufl.TestFunction(V2)
    x = ufl.SpatialCoordinate(msh2)
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    residual_form = (
        (ufl.inner(ufl.grad(u2), ufl.grad(v)) - (k ** 2) * u2 * v - f_expr * v) * ufl.dx
    )
    res_vec = petsc.create_vector(fem.form(residual_form).function_spaces)
    with res_vec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(res_vec, fem.form(residual_form))
    res_norm = res_vec.norm()
    uutwo_l2 = _l2_norm(msh2, u2)

    verification = {
        "mesh_convergence_difference_L2": float(diff_l2),
        "fine_residual_l2_vector_norm": float(res_norm),
        "fine_solution_L2_norm": float(uutwo_l2),
        "coarse_iterations": int(info1["iterations"]),
        "fine_iterations": int(info2["iterations"]),
        "verified_with_resolution_pair": [int(n1), int(n2)],
    }
    return verification


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    out_grid = case_spec["output"]["grid"]

    # Use a reasonably accurate default for k=20 with P2.
    degree = 2
    requested_time_budget = 787.052

    # Adaptive time-accuracy trade-off: choose a conservative high-accuracy mesh,
    # and increase if still far below budget after quick verification on rank 0-relevant wall time.
    mesh_resolution = 96
    if requested_time_budget > 300:
        mesh_resolution = 112

    msh, V, uh, solver_info = _solve_once(comm, mesh_resolution, degree=degree, rtol=1e-9)
    u_grid = _sample_on_grid(msh, uh, out_grid)

    verification = _accuracy_verification(comm, max(24, mesh_resolution // 3), degree)

    elapsed = time.perf_counter() - t0

    # If execution is very fast relative to limit, proactively refine once more.
    if elapsed < 0.15 * requested_time_budget:
        refined_n = min(160, int(round(mesh_resolution * 1.25)))
        msh_r, V_r, uh_r, solver_info_r = _solve_once(comm, refined_n, degree=degree, rtol=1e-9)
        u_grid_r = _sample_on_grid(msh_r, uh_r, out_grid)
        verification_r = _accuracy_verification(comm, max(24, refined_n // 3), degree)

        msh, V, uh = msh_r, V_r, uh_r
        u_grid = u_grid_r
        solver_info = solver_info_r
        verification = verification_r

    solver_info = {
        "mesh_resolution": int(solver_info["mesh_resolution"]),
        "element_degree": int(solver_info["element_degree"]),
        "ksp_type": str(solver_info["ksp_type"]),
        "pc_type": str(solver_info["pc_type"]),
        "rtol": float(solver_info["rtol"]),
        "iterations": int(solver_info["iterations"]),
        "accuracy_verification": verification,
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}


__all__ = ["solve"]


import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_nested(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_grid_points(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    return pts, nx, ny


def _sample_function_on_points(domain, u_func, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(points[i])
            local_cells.append(links[0])

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_points), -1)[:, 0]
        local_vals[np.array(local_ids, dtype=np.int32)] = vals

    if domain.comm.size == 1:
        return local_vals

    send = np.where(np.isnan(local_vals), -np.inf, local_vals)
    recv = np.empty_like(send)
    domain.comm.Allreduce(send, recv, op=MPI.MAX)
    recv[np.isneginf(recv)] = np.nan
    return recv


def _extract_case_parameters(case_spec):
    coeffs = case_spec.get("coefficients", {})
    pde = case_spec.get("pde", {})
    time_spec = case_spec.get("time", {})

    kappa = (
        _get_nested(coeffs, "kappa", default=None)
        if coeffs else None
    )
    if isinstance(kappa, dict):
        kappa = kappa.get("value", None)
    if kappa is None:
        kappa = _get_nested(pde, "coefficients", "kappa", default=0.8)
    if isinstance(kappa, dict):
        kappa = kappa.get("value", 0.8)
    kappa = float(kappa if kappa is not None else 0.8)

    t_end = _get_nested(time_spec, "t_end", default=None)
    if t_end is None:
        t_end = _get_nested(pde, "time", "t_end", default=None)
    if t_end is None:
        t_end = _get_nested(pde, "t_end", default=0.12)
    t_end = float(t_end if t_end is not None else 0.12)

    dt = _get_nested(time_spec, "dt", default=None)
    if dt is None:
        dt = _get_nested(pde, "time", "dt", default=None)
    if dt is None:
        dt = _get_nested(pde, "dt", default=0.02)
    dt = float(dt if dt is not None else 0.02)

    if t_end <= 0:
        t_end = 0.12
    if dt <= 0:
        dt = 0.02

    scheme = _get_nested(time_spec, "scheme", default=None)
    if scheme is None:
        scheme = _get_nested(pde, "time", "scheme", default="backward_euler")
    scheme = str(scheme)

    return kappa, t_end, dt, scheme


def _manufactured_exact(domain, t_end, kappa):
    x = ufl.SpatialCoordinate(domain)
    phi = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    lam = 2.0 * np.pi ** 2 * kappa
    u_exact = (1.0 - ufl.exp(-lam * t_end)) / lam * phi
    f = phi
    return u_exact, f


def _zero_dirichlet_bc(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _solve_heat(mesh_resolution, degree, dt_in, t_end, kappa_value, mode="target",
                ksp_type="cg", pc_type="hypre", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    bc = _zero_dirichlet_bc(V, domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u0 = fem.Function(V)
    u_n.x.array[:] = 0.0
    u0.x.array[:] = 0.0

    x = ufl.SpatialCoordinate(domain)
    if mode == "target":
        source_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
        reference_expr = None
    else:
        reference_expr, source_expr = _manufactured_exact(domain, t_end, kappa_value)

    kappa = fem.Constant(domain, ScalarType(kappa_value))
    dt = min(dt_in, t_end)
    n_steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / n_steps
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * source_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    total_iterations = 0
    start = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_h.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP failed with reason {reason}")
        u_h.x.scatter_forward()
        total_iterations += int(max(0, solver.getIterationNumber()))
        u_n.x.array[:] = u_h.x.array

    runtime = time.perf_counter() - start

    info = {
        "domain": domain,
        "u_final": u_h,
        "u_initial": u0,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "runtime_sec": float(runtime),
    }

    if reference_expr is not None:
        err_form = fem.form((u_h - reference_expr) ** 2 * ufl.dx)
        ref_form = fem.form(reference_expr ** 2 * ufl.dx)
        err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
        ref = np.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
        info["verification"] = {
            "manufactured_l2_error": float(err),
            "manufactured_rel_l2_error": float(err / max(ref, 1e-14)),
            "runtime_sec": float(runtime),
        }
    else:
        info["verification"] = {"runtime_sec": float(runtime)}

    return info


def _run_verification(time_budget):
    configs = [
        (24, 1, 0.01),
        (32, 1, 0.005),
        (40, 1, 0.0025),
        (48, 2, 0.0025),
    ]
    results = []
    elapsed = 0.0
    for n, p, dt in configs:
        try:
            res = _solve_heat(n, p, dt, 0.12, 0.8, mode="manufactured", ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            res = _solve_heat(n, p, dt, 0.12, 0.8, mode="manufactured", ksp_type="preonly", pc_type="lu", rtol=1e-12)
        results.append(res)
        elapsed += res["runtime_sec"]
        if elapsed > 0.3 * time_budget:
            break

    verification = {
        "type": "manufactured_solution",
        "samples": [
            {
                "mesh_resolution": int(r["mesh_resolution"]),
                "element_degree": int(r["element_degree"]),
                "dt": float(r["dt"]),
                "l2_error": float(r["verification"]["manufactured_l2_error"]),
                "rel_l2_error": float(r["verification"]["manufactured_rel_l2_error"]),
                "runtime_sec": float(r["runtime_sec"]),
            }
            for r in results
        ],
    }
    if len(results) >= 2:
        e1 = results[-2]["verification"]["manufactured_l2_error"]
        e2 = results[-1]["verification"]["manufactured_l2_error"]
        h1 = 1.0 / results[-2]["mesh_resolution"]
        h2 = 1.0 / results[-1]["mesh_resolution"]
        if e1 > 0 and e2 > 0 and h1 != h2:
            verification["observed_order_estimate"] = float(np.log(e1 / e2) / np.log(h1 / h2))
    return verification


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    kappa, t_end, dt_suggested, _ = _extract_case_parameters(case_spec)

    time_budget = 18.659
    verification = _run_verification(time_budget)

    candidates = [
        (56, 1, min(dt_suggested, 0.01)),
        (72, 1, min(dt_suggested, 0.005)),
        (88, 1, min(dt_suggested, 0.004)),
        (104, 1, min(dt_suggested, 0.003)),
        (88, 2, min(dt_suggested, 0.004)),
    ]

    chosen = None
    accumulated = sum(s["runtime_sec"] for s in verification["samples"])
    for mesh_resolution, degree, dt in candidates:
        try:
            result = _solve_heat(mesh_resolution, degree, dt, t_end, kappa, mode="target",
                                 ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            result = _solve_heat(mesh_resolution, degree, dt, t_end, kappa, mode="target",
                                 ksp_type="preonly", pc_type="lu", rtol=1e-12)
        chosen = result
        accumulated += result["runtime_sec"]
        if accumulated > 0.8 * time_budget:
            break

    pts, nx, ny = _build_grid_points(grid)
    u_vals = _sample_function_on_points(chosen["domain"], chosen["u_final"], pts)
    u0_vals = _sample_function_on_points(chosen["domain"], chosen["u_initial"], pts)
    if np.isnan(u_vals).any() or np.isnan(u0_vals).any():
        raise RuntimeError("Sampling failed on output grid")

    return {
        "u": u_vals.reshape(ny, nx),
        "u_initial": u0_vals.reshape(ny, nx),
        "solver_info": {
            "mesh_resolution": int(chosen["mesh_resolution"]),
            "element_degree": int(chosen["element_degree"]),
            "ksp_type": str(chosen["ksp_type"]),
            "pc_type": str(chosen["pc_type"]),
            "rtol": float(chosen["rtol"]),
            "iterations": int(chosen["iterations"]),
            "dt": float(chosen["dt"]),
            "n_steps": int(chosen["n_steps"]),
            "time_scheme": str(chosen["time_scheme"]),
            "verification": verification,
        },
    }

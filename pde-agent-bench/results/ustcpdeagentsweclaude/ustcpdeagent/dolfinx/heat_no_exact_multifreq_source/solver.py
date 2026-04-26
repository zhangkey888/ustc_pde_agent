import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_case(case_spec: dict):
    pde = case_spec.get("pde", {})
    coeffs = case_spec.get("coefficients", {})
    grid = _get(case_spec, ["output", "grid"], {}) or {}

    return {
        "t0": float(pde.get("t0", 0.0)),
        "t_end": float(pde.get("t_end", 0.12)),
        "dt_suggested": float(pde.get("dt", 0.02)),
        "scheme": str(pde.get("scheme", "backward_euler")).lower(),
        "kappa": float(coeffs.get("kappa", 1.0)),
        "nx": int(grid.get("nx", 64)),
        "ny": int(grid.get("ny", 64)),
        "bbox": grid.get("bbox", [0.0, 1.0, 0.0, 1.0]),
    }


def _source_ufl(domain):
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    return (
        ufl.sin(5.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1])
        + 0.5 * ufl.sin(9.0 * pi * x[0]) * ufl.sin(7.0 * pi * x[1])
    )


def _boundary_dofs(domain, V):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    return fem.locate_dofs_topological(V, fdim, facets)


def _sample_on_grid(u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = u_func.eval(
            np.array(eval_points, dtype=np.float64),
            np.array(eval_cells, dtype=np.int32),
        )
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.allgather(local_vals)
    merged = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        eps = 1e-12
        pts2 = pts.copy()
        pts2[:, 0] = np.clip(pts2[:, 0], xmin + eps, xmax - eps)
        pts2[:, 1] = np.clip(pts2[:, 1], ymin + eps, ymax - eps)

        candidates2 = geometry.compute_collisions_points(tree, pts2)
        colliding2 = geometry.compute_colliding_cells(domain, candidates2, pts2)
        local_vals2 = np.full(pts.shape[0], np.nan, dtype=np.float64)
        eval_points = []
        eval_cells = []
        eval_ids = []
        for i in np.where(np.isnan(merged))[0]:
            links = colliding2.links(i)
            if len(links) > 0:
                eval_points.append(pts2[i])
                eval_cells.append(links[0])
                eval_ids.append(i)
        if eval_points:
            vals = u_func.eval(
                np.array(eval_points, dtype=np.float64),
                np.array(eval_cells, dtype=np.int32),
            )
            local_vals2[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

        gathered2 = domain.comm.allgather(local_vals2)
        for arr in gathered2:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]

    merged[np.isnan(merged)] = 0.0
    return merged.reshape(ny, nx)


def _run_heat(mesh_resolution, degree, dt, t0, t_end, kappa,
              ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    bc = fem.dirichletbc(ScalarType(0.0), _boundary_dofs(domain, V), V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_h = fem.Function(V)
    u_h.x.array[:] = 0.0

    f_expr = fem.Expression(_source_ufl(domain), V.element.interpolation_points)
    f_fun = fem.Function(V)
    f_fun.interpolate(f_expr)

    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += max(int(its), 0)
        u_n.x.array[:] = u_h.x.array

    return u_h, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }


def solve(case_spec: dict) -> dict:
    params = _parse_case(case_spec)
    t0 = params["t0"]
    t_end = params["t_end"]
    kappa = params["kappa"]
    out_nx = params["nx"]
    out_ny = params["ny"]
    bbox = params["bbox"]

    candidates = [
        {"mesh_resolution": 56, "dt": min(0.01, params["dt_suggested"]), "degree": 1},
        {"mesh_resolution": 72, "dt": 0.0075, "degree": 1},
        {"mesh_resolution": 88, "dt": 0.0060, "degree": 1},
    ]

    start = time.perf_counter()
    chosen_u = None
    chosen_info = None
    prev_grid = None
    verification = {}

    for cand in candidates:
        u_cur, info_cur = _run_heat(
            mesh_resolution=cand["mesh_resolution"],
            degree=cand["degree"],
            dt=cand["dt"],
            t0=t0,
            t_end=t_end,
            kappa=kappa,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1e-10,
        )
        grid_cur = _sample_on_grid(u_cur, out_nx, out_ny, bbox)

        chosen_u = u_cur
        chosen_info = info_cur

        if prev_grid is not None:
            rmse = float(np.sqrt(np.mean((grid_cur - prev_grid) ** 2)))
            verification = {"reference_grid_rmse": rmse}
            if rmse < 2.0e-3:
                break

        prev_grid = grid_cur

        if time.perf_counter() - start > 18.0:
            break

    u_grid = _sample_on_grid(chosen_u, out_nx, out_ny, bbox)
    u_initial = np.zeros((out_ny, out_nx), dtype=np.float64)

    solver_info = dict(chosen_info)
    solver_info.update(verification)

    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial,
    }

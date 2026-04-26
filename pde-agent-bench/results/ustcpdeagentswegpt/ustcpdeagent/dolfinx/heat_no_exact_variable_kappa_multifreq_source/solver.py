import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get_nested(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_time(case_spec):
    pde = case_spec.get("pde", {})
    t0 = _get_nested(case_spec, ["time", "t0"], None)
    if t0 is None:
        t0 = pde.get("t0", 0.0)
    t_end = _get_nested(case_spec, ["time", "t_end"], None)
    if t_end is None:
        t_end = pde.get("t_end", 0.1)
    dt = _get_nested(case_spec, ["time", "dt"], None)
    if dt is None:
        dt = pde.get("dt", 0.02)
    scheme = _get_nested(case_spec, ["time", "scheme"], None)
    if scheme is None:
        scheme = pde.get("scheme", "backward_euler")
    return float(t0), float(t_end), float(dt), str(scheme)


def _default_params(case_spec):
    _, t_end, dt_suggested, _ = _parse_time(case_spec)
    time_budget = 24.719
    nx_out = _get_nested(case_spec, ["output", "grid", "nx"], 64)
    ny_out = _get_nested(case_spec, ["output", "grid", "ny"], 64)
    out_scale = max(nx_out, ny_out)

    if time_budget > 18:
        mesh_n = max(72, min(120, int(max(80, 1.0 * out_scale))))
        degree = 2
        dt = min(dt_suggested, 0.003)
    else:
        mesh_n = 40
        degree = 1
        dt = min(dt_suggested, 0.01)

    if t_end / dt > 120:
        dt = t_end / 120.0

    return mesh_n, degree, dt


def _sample_function_on_grid(domain, uh, grid_spec):
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

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)
        local_vals[np.array(eval_map, dtype=np.int32)] = vals[:, 0]

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            merged[~np.isfinite(merged)] = 0.0
        return merged.reshape((ny, nx))
    return None


def _solve_heat(mesh_n, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )
    u0_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    try:
        solver.getPC().setHYPREType("boomeramg")
    except Exception:
        pass
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=2000)
    solver.setFromOptions()

    initial = fem.Function(V)
    initial.x.array[:] = u_n.x.array[:]
    initial.x.scatter_forward()

    total_iterations = 0
    n_steps = int(round((t_end - t0) / dt))

    for _ in range(n_steps):
        with b.localForm() as bl:
            bl.set(0.0)
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
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()

    info = {
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }
    return domain, initial, u_h, info


def _compute_self_error(case_spec, mesh_n, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    grid = _get_nested(case_spec, ["output", "grid"], {"nx": 48, "ny": 48, "bbox": [0, 1, 0, 1]})

    domain1, _, uh1, _ = _solve_heat(mesh_n, degree, dt, t0, t_end)
    g1 = _sample_function_on_grid(domain1, uh1, grid)

    domain2, _, uh2, _ = _solve_heat(mesh_n, degree, dt / 2.0, t0, t_end)
    g2 = _sample_function_on_grid(domain2, uh2, grid)

    if comm.rank == 0:
        return float(np.linalg.norm(g2 - g1) / max(np.linalg.norm(g2), 1e-14))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    t0, t_end, dt_in, scheme = _parse_time(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    mesh_n, degree, dt = _default_params(case_spec)
    dt = min(dt, dt_in if dt_in > 0 else dt)

    if (t_end - t0) / dt < 10:
        dt = (t_end - t0) / 10.0

    time_budget = comm.bcast(24.719 if comm.rank == 0 else None, root=0)

    verify_dt = min(dt, (t_end - t0) / 12.0)
    if verify_dt > 0:
        try:
            local_t0 = time.perf_counter()
            err_est = _compute_self_error(
                case_spec, min(mesh_n, 48), max(1, min(degree, 2)), verify_dt, t0, t_end
            )
            verify_elapsed = comm.allreduce(time.perf_counter() - local_t0, op=MPI.MAX)
            if comm.rank == 0:
                if verify_elapsed < 0.25 * time_budget:
                    if err_est is None or err_est > 2e-2:
                        mesh_n = min(120, max(mesh_n, 88))
                        degree = max(degree, 2)
                        dt = min(dt, 0.003)
                    else:
                        mesh_n = min(112, max(mesh_n, 84))
                        degree = max(degree, 2)
                        dt = min(dt, 0.003)
                elif err_est is not None and err_est > 5e-2:
                    mesh_n = min(104, max(mesh_n, 80))
                    dt = min(dt, 0.0035)
            mesh_n = comm.bcast(mesh_n, root=0)
            degree = comm.bcast(degree, root=0)
            dt = comm.bcast(dt, root=0)
        except Exception:
            pass

    domain, u_initial_f, u_final_f, solver_info = _solve_heat(mesh_n, degree, dt, t0, t_end)

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, u_final_f, grid_spec)
    u_initial_grid = _sample_function_on_grid(domain, u_initial_f, grid_spec)

    elapsed = comm.allreduce(time.perf_counter() - t_start, op=MPI.MAX)

    if comm.rank == 0:
        solver_info["wall_time_sec"] = float(elapsed)
        return {
            "u": np.asarray(u_grid, dtype=np.float64).reshape((grid_spec["ny"], grid_spec["nx"])),
            "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape((grid_spec["ny"], grid_spec["nx"])),
            "solver_info": solver_info,
        }
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

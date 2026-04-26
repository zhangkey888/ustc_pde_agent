import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_callable_exact(t_value):
    def exact(x):
        return np.exp(-t_value) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return exact


def _make_callable_source(t_value, kappa=1.0):
    coeff = (-1.0 + 2.0 * (np.pi ** 2) * kappa) * np.exp(-t_value)

    def source(x):
        return coeff * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    return source


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        ).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = np.real(vals)

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        return merged.reshape(ny, nx)
    return None


def _compute_l2_error(u_num, u_exact_callable):
    V = u_num.function_space
    u_ex = fem.Function(V)
    u_ex.interpolate(u_exact_callable)
    err_form = fem.form((u_num - u_ex) ** 2 * ufl.dx)
    local = fem.assemble_scalar(err_form)
    global_val = V.mesh.comm.allreduce(local, op=MPI.SUM)
    return float(np.sqrt(global_val))


def _run_heat(n, degree, dt, kappa=1.0, t0=0.0, t_end=0.1, ksp_type="preonly", pc_type="lu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    wall_start = time.perf_counter()

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [int(n), int(n)],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", int(degree)))
    uh = fem.Function(V)
    u_n = fem.Function(V)
    u_bc = fem.Function(V)
    f_fun = fem.Function(V)

    u_n.interpolate(_make_callable_exact(t0))
    uh.x.array[:] = u_n.x.array

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    bc = fem.dirichletbc(u_bc, dofs)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    total_iterations = 0
    n_steps = int(round((t_end - t0) / dt))
    t = t0

    for _ in range(n_steps):
        t += dt
        u_bc.interpolate(_make_callable_exact(t))
        f_fun.interpolate(_make_callable_source(t, kappa=kappa))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array

    l2_error = _compute_l2_error(uh, _make_callable_exact(t_end))
    elapsed = time.perf_counter() - wall_start

    return {
        "domain": domain,
        "u_final": uh,
        "l2_error": l2_error,
        "elapsed": elapsed,
        "iterations": total_iterations,
        "n_steps": n_steps,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "dt": float(dt),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    solve_start = time.perf_counter()

    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.1)))
    suggested_dt = float(time_spec.get("dt", case_spec.get("dt", 0.01)))
    scheme = str(time_spec.get("scheme", "backward_euler"))
    if scheme != "backward_euler":
        scheme = "backward_euler"

    output_grid = case_spec["output"]["grid"]
    total_budget = 1.8
    safety_budget = 1.55
    target_error = 1.11e-3

    candidates = [
        (20, 1, suggested_dt),
        (24, 1, suggested_dt / 2.0),
        (28, 1, suggested_dt / 2.0),
        (24, 2, suggested_dt),
        (32, 2, suggested_dt),
        (36, 2, suggested_dt / 2.0),
        (40, 2, suggested_dt / 2.0),
        (48, 2, suggested_dt / 2.0),
    ]

    best = None
    for n, degree, dt_try in candidates:
        dt_try = float(dt_try)
        n_steps = max(1, int(round((t_end - t0) / dt_try)))
        dt_try = (t_end - t0) / n_steps

        remaining = safety_budget - (time.perf_counter() - solve_start)
        if remaining <= 0.15:
            break

        try:
            result = _run_heat(
                n=n, degree=degree, dt=dt_try, kappa=1.0, t0=t0, t_end=t_end,
                ksp_type="preonly", pc_type="lu", rtol=1e-10,
            )
        except Exception:
            result = _run_heat(
                n=n, degree=degree, dt=dt_try, kappa=1.0, t0=t0, t_end=t_end,
                ksp_type="gmres", pc_type="ilu", rtol=1e-10,
            )

        if result["elapsed"] > max(remaining, 0.2):
            break

        if best is None:
            best = result
        else:
            if result["l2_error"] <= target_error:
                if best["l2_error"] > target_error or result["l2_error"] < best["l2_error"]:
                    best = result
            elif result["l2_error"] < best["l2_error"]:
                best = result

        if (time.perf_counter() - solve_start) > 0.85 * safety_budget:
            break

    while best is not None:
        elapsed_total = time.perf_counter() - solve_start
        if elapsed_total >= 0.78 * safety_budget:
            break
        n_new = min(best["mesh_resolution"] + 8, 56)
        dt_new = best["dt"] / 2.0
        n_steps = max(1, int(round((t_end - t0) / dt_new)))
        dt_new = (t_end - t0) / n_steps
        if n_new == best["mesh_resolution"] and abs(dt_new - best["dt"]) < 1e-14:
            break
        remaining = safety_budget - elapsed_total
        if remaining <= 0.2:
            break
        try:
            cand = _run_heat(
                n=n_new, degree=best["element_degree"], dt=dt_new, kappa=1.0, t0=t0, t_end=t_end,
                ksp_type="preonly", pc_type="lu", rtol=1e-10,
            )
        except Exception:
            break
        if cand["elapsed"] > remaining:
            break
        if cand["l2_error"] <= best["l2_error"]:
            best = cand
        else:
            break

    if best is None:
        best = _run_heat(n=24, degree=2, dt=suggested_dt, kappa=1.0, t0=t0, t_end=t_end, ksp_type="preonly", pc_type="lu", rtol=1e-10)

    domain = best["domain"]
    V = best["u_final"].function_space
    u0_fun = fem.Function(V)
    u0_fun.interpolate(_make_callable_exact(t0))

    u_grid = _sample_function_on_grid(best["u_final"], domain, output_grid)
    u0_grid = _sample_function_on_grid(u0_fun, domain, output_grid)

    result = None
    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": scheme,
            "l2_error": float(best["l2_error"]),
            "wall_time_sec": float(time.perf_counter() - solve_start),
        }
        result = {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
            "u_initial": np.asarray(u0_grid, dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
            "solver_info": solver_info,
        }
    return result


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _extract_time(case_spec):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.01)))
    if dt <= 0:
        dt = 0.01
    return t0, t_end, dt


def _make_exact_ufl(domain, time_value):
    x = ufl.SpatialCoordinate(domain)
    tt = ScalarType(time_value)
    u_exact = np.exp(-float(time_value)) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = 1.0 + 0.5 * ufl.sin(6 * ufl.pi * x[0])
    u_t = -u_exact
    f = u_t - ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, kappa, f


def _interp_function(V, expr):
    fn = fem.Function(V)
    ex = fem.Expression(expr, V.element.interpolation_points)
    fn.interpolate(ex)
    return fn


def _boundary_dofs(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    return fem.locate_dofs_topological(V, fdim, facets)


def _sample_on_grid(u_func, nx, ny, bbox):
    domain = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_map, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            raise RuntimeError(f"Failed to evaluate solution at {len(nan_ids)} output points")
        return out.reshape(ny, nx)
    return None


def _compute_l2_error(domain, uh, u_exact_expr):
    Vh = uh.function_space
    u_ex = _interp_function(Vh, u_exact_expr)
    err_form = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    local = fem.assemble_scalar(err_form)
    global_val = domain.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(max(global_val, 0.0))


def _run_single(case_spec, n, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_n = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u0_expr, _, _ = _make_exact_ufl(domain, t0)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    t_cur = t0 + dt
    uD_expr, kappa_expr, f_expr = _make_exact_ufl(domain, t_cur)
    uD = _interp_function(V, uD_expr)
    bc = fem.dirichletbc(uD, _boundary_dofs(V, domain))

    dt_c = fem.Constant(domain, ScalarType(dt))
    f_fun = _interp_function(V, f_expr)

    a = (u * v + dt_c * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    iterations = 0
    n_steps = int(round((t_end - t0) / dt))
    for step in range(1, n_steps + 1):
        t_cur = t0 + step * dt
        uD_expr, _, f_expr = _make_exact_ufl(domain, t_cur)
        uD.interpolate(fem.Expression(uD_expr, V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

        with b.localForm() as b_loc:
            b_loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += solver.getIterationNumber()
        u_n.x.array[:] = uh.x.array

    uT_expr, _, _ = _make_exact_ufl(domain, t_end)
    l2_error = _compute_l2_error(domain, uh, uT_expr)
    return domain, uh, u_n, l2_error, iterations


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    start = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    t0, t_end, dt_suggested = _extract_time(case_spec)

    candidates = [
        (20, 1, 0.01),
        (24, 1, 0.01),
        (28, 1, 0.008),
        (32, 1, 0.00625),
        (20, 2, 0.01),
        (24, 2, 0.01),
        (28, 2, 0.008),
        (32, 2, 0.00625),
        (40, 2, 0.005),
    ]

    best = None
    feasible = None
    time_budget = 12.292
    safety = 0.94 * time_budget

    for (n, degree, dt) in candidates:
        trial_start = time.perf_counter()
        domain, uh, _, err, iterations = _run_single(case_spec, n, degree, dt, t0, t_end)
        trial_elapsed = time.perf_counter() - trial_start
        elapsed = time.perf_counter() - start

        record = {
            "domain": domain,
            "uh": uh,
            "err": err,
            "iterations": iterations,
            "n": n,
            "degree": degree,
            "dt": dt,
            "n_steps": int(round((t_end - t0) / dt)),
            "trial_elapsed": trial_elapsed,
            "elapsed": elapsed,
        }

        if best is None or err < best["err"]:
            best = record
        if err <= 1.85e-3:
            feasible = record
        if feasible is not None and elapsed >= 0.55 * time_budget:
            best = feasible
            break
        if feasible is not None and elapsed + trial_elapsed > safety:
            best = feasible
            break
        if feasible is None and elapsed + trial_elapsed > safety:
            break

    if feasible is not None and feasible["err"] <= best["err"] + 1e-15:
        best = feasible

    if best is None:
        raise RuntimeError("No solve attempt completed")

    u_initial_grid = None
    if comm.rank == 0:
        pass

    domain = best["domain"]
    V = best["uh"].function_space
    u0_expr, _, _ = _make_exact_ufl(domain, t0)
    u0_fun = _interp_function(V, u0_expr)

    u_grid = _sample_on_grid(best["uh"], nx, ny, bbox)
    u_initial_grid = _sample_on_grid(u0_fun, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(best["n"]),
        "element_degree": int(best["degree"]),
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "l2_error": float(best["err"]),
        "wall_time_sec": float(time.perf_counter() - start),
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.1, "dt": 0.01},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape, out["solver_info"])

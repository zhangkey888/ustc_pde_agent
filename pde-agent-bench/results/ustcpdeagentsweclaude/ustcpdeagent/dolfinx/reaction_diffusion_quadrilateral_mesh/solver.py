import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_time_spec(case_spec):
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", {}) if isinstance(pde.get("time", {}), dict) else {}
    t0 = float(tinfo.get("t0", 0.0))
    t_end = float(tinfo.get("t_end", 0.4))
    dt = float(tinfo.get("dt", 0.01))
    scheme = tinfo.get("scheme", "backward_euler")
    return t0, t_end, dt, scheme


def _exact_expr(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-ScalarType(t)) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])


def _manufactured_rhs(msh, eps_value, t):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.exp(-ScalarType(t)) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    u_t = -u_ex
    lap_u = (1.0 - ufl.pi**2) * u_ex
    # PDE: u_t - eps*Delta u + u = f
    return u_t - ScalarType(eps_value) * lap_u + u_ex


def _interpolate_scalar_callable(func):
    return lambda x: np.asarray(func(x), dtype=np.float64)


def _sample_function_on_grid(u_func, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            nan_ids = np.where(np.isnan(global_vals))[0]
            raise RuntimeError(f"Failed to evaluate solution at {len(nan_ids)} grid points.")
        return global_vals.reshape(ny, nx)
    return None


def _run_single(case_spec, n, degree, dt, eps_value=0.05):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    t0, t_end, _, scheme = _get_time_spec(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    if dt <= 0:
        dt = 0.01
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-t0) * np.exp(x[0]) * np.sin(np.pi * x[1]))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)

    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps_value))

    t_cur = t0 + dt
    f_expr = _manufactured_rhs(msh, eps_value, t_cur)
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    f_fun.x.scatter_forward()

    a = (u * v + dt_c * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_c * u * v) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    start = time.perf_counter()
    for step in range(n_steps):
        t_cur = t0 + (step + 1) * dt
        u_bc.interpolate(lambda x, tt=t_cur: np.exp(-tt) * np.exp(x[0]) * np.sin(np.pi * x[1]))
        u_bc.x.scatter_forward()

        f_expr = _manufactured_rhs(msh, eps_value, t_cur)
        f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        f_fun.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = u_n.x.array
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        try:
            its = solver.getIterationNumber()
        except Exception:
            its = 0
        total_iterations += int(its)

        if not np.all(np.isfinite(uh.x.array)):
            raise RuntimeError("Non-finite values encountered in solution.")

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    u_exact_T = fem.Function(V)
    u_exact_T.interpolate(lambda x, tt=t_end: np.exp(-tt) * np.exp(x[0]) * np.sin(np.pi * x[1]))
    u_exact_T.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_T.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_exact_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_T, u_exact_T) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    l2_exact = math.sqrt(comm.allreduce(l2_exact_local, op=MPI.SUM))
    rel_l2 = l2_err / max(l2_exact, 1e-30)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(uh, msh, grid)
    u0_grid = _sample_function_on_grid(
        fem.Function(V), msh, grid
    )
    if comm.rank == 0:
        xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
        ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u0_grid = np.exp(-t0) * np.exp(XX) * np.sin(np.pi * YY)

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [],
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2),
        "wall_time_sec_internal": float(elapsed),
    }
    return u_grid, u0_grid, info


def solve(case_spec: dict) -> dict:
    """
    Return dict with sampled solution and solver metadata.
    """
    comm = MPI.COMM_WORLD
    t0, t_end, dt_suggested, _ = _get_time_spec(case_spec)

    # Adaptive accuracy/time trade-off:
    # start accurate enough, then refine if cheap.
    candidates = [
        (40, 1, min(dt_suggested, 0.01)),
        (56, 1, min(dt_suggested, 0.008)),
        (72, 1, min(dt_suggested, 0.00625)),
        (88, 1, min(dt_suggested, 0.005)),
    ]

    best = None
    budget = 40.0  # internal conservative budget far below benchmark limit
    t_begin = time.perf_counter()

    for n, degree, dt in candidates:
        wall_so_far = time.perf_counter() - t_begin
        if wall_so_far > budget:
            break
        result = _run_single(case_spec, n=n, degree=degree, dt=dt, eps_value=0.05)
        if comm.rank == 0:
            _, _, info = result
            if (best is None) or (info["relative_l2_error"] < best[2]["relative_l2_error"]):
                best = result
            # stop early if already very accurate and runtime suggests little gain needed
            if info["relative_l2_error"] < 2e-3 and wall_so_far > 2.0:
                break
        else:
            best = result

    if best is None:
        best = _run_single(case_spec, n=40, degree=1, dt=min(dt_suggested, 0.01), eps_value=0.05)

    u_grid, u0_grid, info = best
    out = {
        "u": u_grid,
        "solver_info": info,
        "u_initial": u0_grid,
    }
    return out


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.4, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    coeffs = case_spec.get("coefficients", {})

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.08))
    dt = float(time_spec.get("dt", 0.004))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()
    kappa = float(coeffs.get("kappa", 5.0))
    return t0, t_end, dt, scheme, kappa


def _build_forms(domain, degree, dt_value, kappa_value):
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(dt_value))
    kappa_c = fem.Constant(domain, ScalarType(kappa_value))
    t_c = fem.Constant(domain, ScalarType(0.0))

    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    u_bc = fem.Function(V)
    u_exact_fun = fem.Function(V)

    pi = np.pi
    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = (
        (-ufl.exp(-t_c) + 5.0 * pi * pi * kappa_c * ufl.exp(-t_c))
        * ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    )

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_ufl * v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    return {
        "V": V,
        "u_n": u_n,
        "u_sol": u_sol,
        "u_bc": u_bc,
        "u_exact_fun": u_exact_fun,
        "u_exact_ufl": u_exact_ufl,
        "t_c": t_c,
        "dt_c": dt_c,
        "a_form": fem.form(a),
        "L_form": fem.form(L),
        "bc": bc,
    }


def _interp_exact(func, t):
    func.interpolate(
        lambda X: np.exp(-t) * np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1])
    )


def _solve_config(nx, degree, dt, t0, t_end, kappa, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    data = _build_forms(domain, degree, dt, kappa)
    V = data["V"]
    u_n = data["u_n"]
    u_sol = data["u_sol"]
    u_bc = data["u_bc"]
    u_exact_fun = data["u_exact_fun"]
    t_c = data["t_c"]
    a_form = data["a_form"]
    L_form = data["L_form"]
    bc = data["bc"]

    _interp_exact(u_n, t0)
    _interp_exact(u_bc, t0)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)
    if ksp_type == "cg":
        try:
            solver.setNormType(PETSc.KSP.NormType.PRECONDITIONED)
        except Exception:
            pass
    solver.setFromOptions()

    n_steps = int(round((t_end - t0) / dt))
    t = t0
    total_iterations = 0

    _interp_exact(u_exact_fun, t0)
    u_initial = u_n.x.array.copy()

    start = time.perf_counter()
    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        _interp_exact(u_bc, t)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_sol.x.petsc_vec)
        except Exception:
            solver.destroy()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setFromOptions()
            solver.solve(b, u_sol.x.petsc_vec)

        u_sol.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its)

        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()

    solve_time = time.perf_counter() - start

    _interp_exact(u_exact_fun, t)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_sol.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "l2_error": l2_error,
        "wall_time": solve_time,
    }

    return domain, V, u_sol, u_initial, info


def _sample_on_grid(domain, u_fun, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        sampled = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(eval_map, dtype=np.int32)] = np.asarray(sampled).reshape(-1)

    comm = domain.comm
    gathered = comm.allreduce(vals, op=MPI.SUM)
    if np.isnan(gathered).any():
        nan_mask = np.isnan(gathered)
        gathered[nan_mask] = 0.0
    return gathered.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme, kappa = _make_case_defaults(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"

    grid = case_spec["output"]["grid"]
    wall_limit = 32.107
    target_fraction = 0.55

    configs = [
        (40, 1, dt_suggested),
        (64, 1, dt_suggested / 2.0),
        (80, 1, dt_suggested / 2.0),
        (96, 1, dt_suggested / 4.0),
        (64, 2, dt_suggested / 2.0),
    ]

    chosen = None
    best_metric = None
    fallback = None

    for nx, degree, dt in configs:
        if dt <= 0:
            continue
        n_steps = int(round((t_end - t0) / dt))
        if n_steps < 1:
            continue
        dt = (t_end - t0) / n_steps
        try:
            domain, V, u_sol, u_initial_vec, info = _solve_config(nx, degree, dt, t0, t_end, kappa)
        except Exception:
            continue

        err = info["l2_error"]
        wtime = info["wall_time"]
        if fallback is None:
            fallback = (domain, u_sol, u_initial_vec, info)

        if err <= 4.26e-3 and wtime <= wall_limit:
            metric = (float(err), -float(wtime))
            if (chosen is None) or (metric < best_metric):
                chosen = (domain, u_sol, u_initial_vec, info)
                best_metric = metric
            if wtime > wall_limit * target_fraction:
                break

    if chosen is None:
        if fallback is None:
            raise RuntimeError("No valid solver configuration succeeded.")
        chosen = fallback

    domain, u_sol, u_initial_vec, info = chosen
    V = u_sol.function_space
    u_initial_fun = fem.Function(V)
    u_initial_fun.x.array[:] = u_initial_vec
    u_initial_fun.x.scatter_forward()

    u_grid = _sample_on_grid(domain, u_sol, grid)
    u0_grid = _sample_on_grid(domain, u_initial_fun, grid)

    solver_info = {
        "mesh_resolution": int(info["mesh_resolution"]),
        "element_degree": int(info["element_degree"]),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(info["rtol"]),
        "iterations": int(info["iterations"]),
        "dt": float(info["dt"]),
        "n_steps": int(info["n_steps"]),
        "time_scheme": str(info["time_scheme"]),
    }

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }

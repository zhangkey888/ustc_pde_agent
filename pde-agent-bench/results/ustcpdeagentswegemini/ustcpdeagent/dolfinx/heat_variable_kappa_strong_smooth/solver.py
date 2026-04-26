import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _parse_case_spec(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    output = case_spec.get("output", {})
    grid = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    return {
        "t0": float(time_spec.get("t0", 0.0)),
        "t_end": float(time_spec.get("t_end", 0.1)),
        "dt_suggested": float(time_spec.get("dt", 0.01)),
        "scheme": str(time_spec.get("scheme", "backward_euler")),
        "nx_out": int(grid.get("nx", 64)),
        "ny_out": int(grid.get("ny", 64)),
        "bbox": grid.get("bbox", [0.0, 1.0, 0.0, 1.0]),
    }


def _u_exact_expr(x, t):
    pi = np.pi
    return np.exp(-t) * np.sin(3 * pi * x[0]) * np.sin(2 * pi * x[1])


def _interp_exact(u_fun, t):
    u_fun.interpolate(lambda x: _u_exact_expr(x, t))


def _build_forms(msh, V, dt):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    kappa = 1 + 0.8 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])

    sinx = ufl.sin(3 * pi * x[0])
    siny = ufl.sin(2 * pi * x[1])
    cosx = ufl.cos(3 * pi * x[0])
    cosy = ufl.cos(2 * pi * x[1])

    uamp = sinx * siny
    ux = 3 * pi * cosx * siny
    uy = 2 * pi * sinx * cosy
    kx = 1.6 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    ky = 1.6 * pi * ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    lap_uamp = -(13 * pi * pi) * uamp
    gradk_gradu = kx * ux + ky * uy
    div_k_grad_uamp = gradk_gradu + kappa * lap_uamp

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    t_np1 = fem.Constant(msh, ScalarType(0.0))

    f_np1 = -ufl.exp(-t_np1) * uamp - ufl.exp(-t_np1) * div_k_grad_uamp

    a = (u * v + dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_np1 * v) * ufl.dx
    return u_n, t_np1, a, L


def _sample_function_on_grid(msh, u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_local, cells_local, idx_local = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_local.append(pts[i])
            cells_local.append(links[0])
            idx_local.append(i)

    if points_local:
        vals = u_func.eval(np.array(points_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if msh.comm.size > 1:
        send = np.nan_to_num(values, nan=0.0)
        recv = np.zeros_like(send)
        msh.comm.Allreduce(send, recv, op=MPI.SUM)
        values = recv

    return values.reshape(ny, nx)


def _solve_once(mesh_resolution, degree, dt, t0, t_end, scheme):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u_n, t_np1, a, L = _build_forms(msh, V, dt)

    _interp_exact(u_n, t0)
    u_sol = fem.Function(V)
    _interp_exact(u_sol, t0)

    uD = fem.Function(V)
    _interp_exact(uD, t0 + dt)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        t_np1.value = ScalarType(t)
        _interp_exact(uD, t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_sol.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, u_sol.x.petsc_vec)

        u_sol.x.scatter_forward()
        total_iterations += max(solver.getIterationNumber(), 0)
        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()

    actual_t = t0 + n_steps * dt

    u_exact_T = fem.Function(V)
    _interp_exact(u_exact_T, actual_t)
    err = fem.Function(V)
    err.x.array[:] = u_sol.x.array - u_exact_T.x.array
    err.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    elapsed = time.perf_counter() - start

    info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": scheme,
        "l2_error": float(l2_error),
        "wall_time": float(elapsed),
        "t_final": float(actual_t),
    }
    return msh, V, u_sol, info


def solve(case_spec: dict) -> dict:
    spec = _parse_case_spec(case_spec)
    t0 = spec["t0"]
    t_end = spec["t_end"]
    scheme = spec["scheme"]
    nx_out = spec["nx_out"]
    ny_out = spec["ny_out"]
    bbox = spec["bbox"]
    dt0 = spec["dt_suggested"]

    candidates = [
        (40, 1, min(dt0, 0.01)),
        (56, 1, min(dt0, 0.005)),
        (64, 2, min(dt0, 0.005)),
        (80, 2, min(dt0, 0.0025)),
        (96, 2, min(dt0, 0.00125)),
    ]

    best = None
    t_start = time.perf_counter()
    for mesh_resolution, degree, dt in candidates:
        if time.perf_counter() - t_start > 20.0:
            break
        try:
            result = _solve_once(mesh_resolution, degree, dt, t0, t_end, scheme)
        except Exception:
            continue
        if best is None or result[3]["l2_error"] < best[3]["l2_error"]:
            best = result
        if result[3]["l2_error"] <= 6.63e-03 and (time.perf_counter() - t_start) > 10.0:
            break

    if best is None:
        raise RuntimeError("Failed to compute heat equation solution.")

    msh, V, u_sol, info = best

    u0_f = fem.Function(V)
    _interp_exact(u0_f, t0)

    u_grid = _sample_function_on_grid(msh, u_sol, nx_out, ny_out, bbox)
    u_initial = _sample_function_on_grid(msh, u0_f, nx_out, ny_out, bbox)

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
        "accuracy_verification": {
            "l2_error_against_manufactured": float(info["l2_error"]),
            "wall_time_sec": float(info["wall_time"]),
            "t_final": float(info["t_final"]),
        },
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.asarray(u_initial, dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

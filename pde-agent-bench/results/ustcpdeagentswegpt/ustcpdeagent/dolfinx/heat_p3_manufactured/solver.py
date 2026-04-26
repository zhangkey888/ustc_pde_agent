import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u_expr(x, t):
    return np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _build_problem(nx, degree, dt, t_end, kappa=1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_exact_fun = fem.Function(V)
    bc_fun = fem.Function(V)

    u_n.interpolate(lambda X: _exact_u_expr(X, 0.0))
    u_exact_fun.interpolate(lambda X: _exact_u_expr(X, 0.0))
    bc_fun.interpolate(lambda X: _exact_u_expr(X, 0.0))

    f_expr = (
        (-ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]))
        + 5.0 * ufl.pi * ufl.pi * kappa_c * ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    )

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(bc_fun, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)
    solver.setFromOptions()

    return {
        "domain": domain,
        "V": V,
        "t_c": t_c,
        "dt_c": dt_c,
        "u_n": u_n,
        "u_h": u_h,
        "u_exact_fun": u_exact_fun,
        "bc_fun": bc_fun,
        "bc": bc,
        "a_form": a_form,
        "L_form": L_form,
        "A": A,
        "b": b,
        "solver": solver,
        "degree": degree,
        "nx": nx,
        "dt": dt,
        "t_end": t_end,
    }


def _run_case(nx, degree, dt, t_end):
    prob = _build_problem(nx, degree, dt, t_end)
    domain = prob["domain"]
    V = prob["V"]
    t_c = prob["t_c"]
    u_n = prob["u_n"]
    u_h = prob["u_h"]
    u_exact_fun = prob["u_exact_fun"]
    bc_fun = prob["bc_fun"]
    bc = prob["bc"]
    a_form = prob["a_form"]
    L_form = prob["L_form"]
    b = prob["b"]
    solver = prob["solver"]

    n_steps = int(round(t_end / dt))
    t = 0.0
    total_iterations = 0

    t_start = time.perf_counter()
    for _ in range(n_steps):
        t = round(t + dt, 12)
        t_c.value = ScalarType(t)
        bc_fun.interpolate(lambda X, tt=t: _exact_u_expr(X, tt))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its)

        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - t_start

    u_exact_fun.interpolate(lambda X, tt=t: _exact_u_expr(X, tt))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((err_fun**2) * ufl.dx))
    l2_err = math.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))

    return {
        "problem": prob,
        "u": u_h,
        "u_initial": _sample_on_grid(u_exact_fun, {
            "nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]
        }) if False else None,
        "l2_error": l2_err,
        "wall_time": elapsed,
        "iterations": total_iterations,
        "n_steps": n_steps,
        "t_final": t,
    }


def _sample_on_grid(u_func, grid_spec):
    domain = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(map_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allreduce(np.nan_to_num(values, nan=0.0), op=MPI.SUM)
    return gathered.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", 0.08))
    dt_suggested = float(case_spec.get("pde", {}).get("time", {}).get("dt", 0.008))
    grid = case_spec["output"]["grid"]

    candidates = [
        (24, 2, min(dt_suggested, 0.004)),
        (32, 2, min(dt_suggested, 0.004)),
        (40, 2, min(dt_suggested, 0.002)),
        (48, 2, min(dt_suggested, 0.002)),
        (56, 3, min(dt_suggested, 0.002)),
    ]

    target_error = 7.97e-04
    time_limit = 24.471
    start_total = time.perf_counter()

    best = None
    for nx, degree, dt in candidates:
        remaining = time_limit - (time.perf_counter() - start_total)
        if remaining <= 1.0:
            break
        try:
            result = _run_case(nx, degree, dt, t_end)
        except Exception:
            continue

        best = {
            "result": result,
            "nx": nx,
            "degree": degree,
            "dt": dt,
        }

        if result["l2_error"] <= target_error:
            if result["wall_time"] > 0.35 * time_limit:
                break
            else:
                continue

    if best is None:
        raise RuntimeError("Failed to solve heat equation with available configurations.")

    result = best["result"]
    u_grid = _sample_on_grid(result["u"], grid)

    prob = result["problem"]
    V = prob["V"]
    u0_fun = fem.Function(V)
    u0_fun.interpolate(lambda X: _exact_u_expr(X, 0.0))
    u0_grid = _sample_on_grid(u0_fun, grid)

    solver = prob["solver"]
    ksp_type = solver.getType()
    pc_type = solver.getPC().getType()
    rtol = solver.getTolerances()[0]

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(best["nx"]),
            "element_degree": int(best["degree"]),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(result["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(result["n_steps"]),
            "time_scheme": "backward_euler",
            "l2_error": float(result["l2_error"]),
            "wall_time": float(result["wall_time"]),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.08, "dt": 0.008}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

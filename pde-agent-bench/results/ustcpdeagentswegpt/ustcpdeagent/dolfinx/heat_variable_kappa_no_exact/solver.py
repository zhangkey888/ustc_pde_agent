import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_expr_callable(expr: str):
    code = compile(expr, "<expr>", "eval")
    safe_dict = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "np": np,
    }

    def f(x):
        xv = x[0]
        yv = x[1] if x.shape[0] > 1 else 0.0
        zv = x[2] if x.shape[0] > 2 else 0.0
        return np.asarray(eval(code, {"__builtins__": {}}, {"x": xv, "y": yv, "z": zv, **safe_dict}), dtype=np.float64)

    return f


def _source_callable():
    def f(x):
        return 1.0 + np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])
    return f


def _initial_callable():
    def f(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return f


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2d = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts2d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2d)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            final[np.isnan(final)] = 0.0
        final = final.reshape(ny, nx)
    else:
        final = None

    final = comm.bcast(final, root=0)
    return final


def _run_heat(case_spec, mesh_resolution=48, degree=1, dt=0.01, t_end=0.1, verify_grid=None):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa_spec = case_spec.get("coefficients", {}).get("kappa", {"type": "expr", "expr": "1.0"})
    if kappa_spec.get("type", "") == "expr":
        kappa_expr = kappa_spec.get("expr", "1.0").replace("x", "x[0]").replace("y", "x[1]")
        kappa_ufl = eval(
            compile(kappa_expr, "<kappa_ufl>", "eval"),
            {"__builtins__": {}},
            {"x": x, "sin": ufl.sin, "cos": ufl.cos, "tan": ufl.tan, "exp": ufl.exp, "sqrt": ufl.sqrt, "pi": ufl.pi},
        )
        kappa_callable = _make_expr_callable(kappa_spec.get("expr", "1.0"))
    else:
        val = float(kappa_spec.get("value", 1.0))
        kappa_ufl = ScalarType(val)
        kappa_callable = lambda xx, vv=val: np.full(xx.shape[1], vv, dtype=np.float64)

    f_callable = _source_callable()
    u0_callable = _initial_callable()

    u_n = fem.Function(V)
    u_n.interpolate(u0_callable)
    uh = fem.Function(V)
    f_fun = fem.Function(V)
    f_fun.interpolate(f_callable)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    a = (u * v + dt_c * ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    n_steps = int(round(t_end / dt))
    iterations = 0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its == 0 and solver.getType().lower() == "preonly":
            its = 1
        iterations += int(its)
        u_n.x.array[:] = uh.x.array

    solve_time = time.perf_counter() - t0

    if verify_grid is None:
        verify_grid = {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}
    u_grid = _sample_function_on_grid(domain, uh, verify_grid)
    u_init_grid = _sample_function_on_grid(domain, fem.Function(V), verify_grid)
    if comm.rank == 0:
        Xs = np.linspace(verify_grid["bbox"][0], verify_grid["bbox"][1], verify_grid["nx"])
        Ys = np.linspace(verify_grid["bbox"][2], verify_grid["bbox"][3], verify_grid["ny"])
        XX, YY = np.meshgrid(Xs, Ys, indexing="xy")
        u_init_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    u_init_grid = comm.bcast(u_init_grid, root=0)

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "u_grid": u_grid,
        "u_initial_grid": u_init_grid,
        "iterations": iterations,
        "solve_time": solve_time,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-9,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    output_grid = case_spec["output"]["grid"]

    t0 = float(case_spec.get("time", {}).get("t0", case_spec.get("pde", {}).get("time", {}).get("t0", 0.0)))
    t_end = float(case_spec.get("time", {}).get("t_end", case_spec.get("pde", {}).get("time", {}).get("t_end", 0.1)))
    dt_suggested = float(case_spec.get("time", {}).get("dt", case_spec.get("pde", {}).get("time", {}).get("dt", 0.02)))
    horizon = t_end - t0
    if horizon <= 0:
        horizon = 0.1
        t_end = t0 + horizon

    mesh_resolution = 56
    degree = 1
    dt = min(dt_suggested, 0.01)
    dt = horizon / max(1, int(round(horizon / dt)))

    coarse = _run_heat(case_spec, mesh_resolution=mesh_resolution, degree=degree, dt=dt, t_end=horizon, verify_grid=output_grid)

    # Accuracy verification / adaptive improvement:
    # compare against a finer temporal solve on same mesh if affordable.
    do_refine = coarse["solve_time"] < 6.0
    temporal_indicator = None

    if do_refine:
        fine_dt = dt / 2.0
        fine_dt = horizon / max(1, int(round(horizon / fine_dt)))
        fine = _run_heat(case_spec, mesh_resolution=mesh_resolution, degree=degree, dt=fine_dt, t_end=horizon, verify_grid=output_grid)
        if comm.rank == 0:
            temporal_indicator = float(np.linalg.norm(fine["u_grid"] - coarse["u_grid"]) / np.sqrt(fine["u_grid"].size))
        temporal_indicator = comm.bcast(temporal_indicator, root=0)

        # If plenty of time remains, use the finer result as final.
        if coarse["solve_time"] + fine["solve_time"] < 14.0:
            result = fine
        else:
            result = coarse
    else:
        result = coarse
        temporal_indicator = None

    # If still cheap, improve spatially as well.
    if result["solve_time"] < 4.0:
        refined_mesh = min(80, max(result["mesh_resolution"] + 16, int(result["mesh_resolution"] * 1.25)))
        trial = _run_heat(case_spec, mesh_resolution=refined_mesh, degree=degree, dt=result["dt"], t_end=horizon, verify_grid=output_grid)
        if result["solve_time"] + trial["solve_time"] < 14.5:
            if comm.rank == 0:
                temporal_indicator = float(np.linalg.norm(trial["u_grid"] - result["u_grid"]) / np.sqrt(trial["u_grid"].size))
            temporal_indicator = comm.bcast(temporal_indicator, root=0)
            result = trial

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": str(result["time_scheme"]),
    }
    if temporal_indicator is not None:
        solver_info["accuracy_verification"] = {
            "type": "self_temporal_or_resolution_check",
            "indicator_l2_grid": float(temporal_indicator),
        }

    return {
        "u": result["u_grid"].astype(np.float64),
        "u_initial": result["u_initial_grid"].astype(np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "coefficients": {"kappa": {"type": "expr", "expr": "1 + 0.6*sin(2*pi*x)*sin(2*pi*y)"}},
        "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

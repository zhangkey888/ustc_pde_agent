import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _expr_u0(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
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
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        return merged.reshape(ny, nx)
    return None


def _solve_heat_once(mesh_resolution, degree, dt, t_end, ksp_type="cg", pc_type="hypre", rtol=1e-8):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u_n = fem.Function(V)
    u_n.interpolate(_expr_u0)
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    if solver.getType().lower() in ("cg", "minres"):
        try:
            A.setOption(PETSc.Mat.Option.SPD, True)
        except Exception:
            pass

    n_steps = int(round(t_end / dt))
    iterations = 0

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=rtol)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its is not None and its >= 0:
            iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "iterations": iterations,
        "n_steps": n_steps,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "dt": dt,
    }


def _estimate_self_error(coarse_grid, fine_grid):
    return float(np.linalg.norm(fine_grid - coarse_grid) / math.sqrt(fine_grid.size))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_in = float(time_spec.get("dt", 0.02))
    output_grid = case_spec["output"]["grid"]

    time_budget = 25.645
    start = time.perf_counter()

    degree = 1
    candidates = [
        (48, min(dt_in, 0.01)),
        (64, min(dt_in, 0.005)),
        (80, min(dt_in, 0.005)),
        (96, min(dt_in, 0.004)),
        (112, min(dt_in, 0.003125)),
        (128, min(dt_in, 0.0025)),
        (144, min(dt_in, 0.0020)),
    ]

    best = None
    prev_grid = None
    last_runtime = None

    for i, (mesh_res, dt) in enumerate(candidates):
        if time.perf_counter() - start > 0.92 * time_budget:
            break

        run_start = time.perf_counter()
        result = _solve_heat_once(mesh_res, degree, dt, t_end - t0, ksp_type="cg", pc_type="hypre", rtol=1e-8)
        u_grid = _sample_function_on_grid(result["u"], result["domain"], output_grid)

        u0_fun = fem.Function(result["V"])
        u0_fun.interpolate(_expr_u0)
        u0_fun.x.scatter_forward()
        u0_grid = _sample_function_on_grid(u0_fun, result["domain"], output_grid)

        run_time = time.perf_counter() - run_start

        if rank == 0:
            result["u_grid"] = u_grid
            result["u0_grid"] = u0_grid
            result["self_error"] = None if prev_grid is None else _estimate_self_error(prev_grid, u_grid)
            prev_grid = u_grid.copy()
        else:
            result["u_grid"] = None
            result["u0_grid"] = None
            result["self_error"] = None

        best = result
        last_runtime = run_time

        if i < len(candidates) - 1:
            projected = (time.perf_counter() - start) + 1.35 * max(run_time, 1e-6)
            if projected > 0.97 * time_budget:
                break

    if best is None:
        best = _solve_heat_once(48, degree, min(dt_in, 0.01), t_end - t0, ksp_type="cg", pc_type="hypre", rtol=1e-8)
        best["u_grid"] = _sample_function_on_grid(best["u"], best["domain"], output_grid)
        u0_fun = fem.Function(best["V"])
        u0_fun.interpolate(_expr_u0)
        u0_fun.x.scatter_forward()
        best["u0_grid"] = _sample_function_on_grid(u0_fun, best["domain"], output_grid)

    if rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": "backward_euler",
            "accuracy_verification": {
                "type": "self_convergence_on_output_grid",
                "estimated_change_vs_previous_candidate": None if best.get("self_error") is None else float(best["self_error"]),
                "wall_time_last_candidate": None if last_runtime is None else float(last_runtime),
            },
        }
        return {
            "u": np.asarray(best["u_grid"], dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
            "u_initial": np.asarray(best["u0_grid"], dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
            "solver_info": solver_info,
        }

    return {"u": None, "u_initial": None, "solver_info": {}}

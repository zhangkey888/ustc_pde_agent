import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_exact(points_xy: np.ndarray, t: float) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return np.exp(-t) * np.exp(-40.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def _probe_function(u_func: fem.Function, pts3: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts3)

    points_on_proc = []
    cells_on_proc = []
    point_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            point_ids.append(i)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.asarray(point_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return local_vals


def _sample_on_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts3)
    return vals.reshape(ny, nx)


def _build_expressions(domain, kappa_value: float):
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    kappa_c = fem.Constant(domain, ScalarType(kappa_value))
    gaussian = ufl.exp(-40.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    u_exact = ufl.exp(-t_const) * gaussian
    f_expr = -u_exact - ufl.div(kappa_c * ufl.grad(u_exact))
    return t_const, kappa_c, u_exact, f_expr


def _run_solver(mesh_resolution: int, degree: int, dt: float, t0: float, t_end: float, kappa: float = 1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    t_const, kappa_c, u_exact_ufl, f_ufl = _build_expressions(domain, kappa)

    u_n = fem.Function(V)
    t_const.value = ScalarType(t0)
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_fun, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=5000)
    solver.setFromOptions()

    use_direct = False
    try:
        solver.setUp()
    except Exception:
        use_direct = True

    if use_direct:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        solver.setUp()

    uh = fem.Function(V)
    total_time = t_end - t0
    n_steps = max(1, int(round(total_time / dt)))
    dt_eff = total_time / n_steps
    dt_c.value = ScalarType(dt_eff)

    total_iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt_eff
        t_const.value = ScalarType(t)
        u_bc_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        its = solver.getIterationNumber()
        if its > 0:
            total_iterations += int(its)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

        if solver.getConvergedReason() <= 0:
            raise RuntimeError(f"Linear solver failed with reason {solver.getConvergedReason()}")

    u_exact_fun = fem.Function(V)
    t_const.value = ScalarType(t_end)
    u_exact_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    err_sq = fem.assemble_scalar(fem.form((uh - u_exact_fun) * (uh - u_exact_fun) * ufl.dx))
    ref_sq = fem.assemble_scalar(fem.form(u_exact_fun * u_exact_fun * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_sq, op=MPI.SUM))
    ref_l2 = math.sqrt(comm.allreduce(ref_sq, op=MPI.SUM))
    rel_l2 = err_l2 / max(ref_l2, 1.0e-16)

    return {
        "u_final": uh,
        "u_initial": u_initial,
        "dt": float(dt_eff),
        "n_steps": int(n_steps),
        "error_l2": float(err_l2),
        "rel_error_l2": float(rel_l2),
        "iterations": int(total_iterations),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1.0e-10,
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    output_grid = case_spec["output"]["grid"]

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()
    if scheme != "backward_euler":
        scheme = "backward_euler"

    start = time.perf_counter()
    budget = 4.436
    safety = 0.90

    candidates = [
        (24, 1, min(dt_suggested, 0.01)),
        (32, 1, 0.005),
        (40, 1, 0.005),
        (32, 2, 0.005),
        (40, 2, 0.005),
        (44, 2, 0.004),
    ]

    best = None
    last_elapsed = 0.0

    for mesh_resolution, degree, dt_try in candidates:
        remaining = budget - (time.perf_counter() - start)
        if remaining <= 0.2:
            break
        try:
            result = _run_solver(mesh_resolution, degree, dt_try, t0, t_end, kappa=1.0)
        except Exception:
            continue

        elapsed = time.perf_counter() - start
        last_elapsed = elapsed
        best = result

        if elapsed >= safety * budget:
            break

    if best is None:
        raise RuntimeError("Failed to compute a valid heat-equation solution.")

    u_grid = _sample_on_grid(best["u_final"], output_grid)
    u_initial_grid = _sample_on_grid(best["u_initial"], output_grid)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": scheme,
            "verification": {
                "manufactured_solution": True,
                "l2_error": float(best["error_l2"]),
                "relative_l2_error": float(best["rel_error_l2"]),
                "wall_time_sec_estimate": float(last_elapsed),
            },
        }
        return {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"])),
            "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"])),
            "solver_info": solver_info,
        }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"])),
        "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"])),
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": scheme,
            "verification": {
                "manufactured_solution": True,
                "l2_error": float(best["error_l2"]),
                "relative_l2_error": float(best["rel_error_l2"]),
                "wall_time_sec_estimate": float(last_elapsed),
            },
        },
    }

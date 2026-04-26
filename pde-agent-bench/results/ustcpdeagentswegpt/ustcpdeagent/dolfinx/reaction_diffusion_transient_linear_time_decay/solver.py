import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _parse_case(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.4))
    dt_in = time_spec.get("dt", 0.02)
    dt = float(dt_in if dt_in is not None else 0.02)
    scheme = time_spec.get("scheme", "backward_euler")

    coeffs = pde.get("coefficients", case_spec.get("coefficients", {}))
    epsilon = coeffs.get("epsilon", case_spec.get("epsilon", 0.1))
    epsilon = float(epsilon if epsilon is not None else 0.1)

    return t0, t_end, dt, scheme, epsilon


def _exact_u_expr(x, t):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(x, t, epsilon):
    u_exact = _exact_u_expr(x, t)
    u_t = -u_exact
    lap_u = -2.0 * (ufl.pi ** 2) * u_exact
    return u_t - epsilon * lap_u + u_exact


def _sample_on_grid(domain, u_fun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            # Fill any unresolved points by exact boundary-consistent value fallback
            out[np.isnan(out)] = 0.0
        out = out.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_once(case_spec, nx_mesh=48, degree=1, dt_override=None):
    comm = MPI.COMM_WORLD
    t0, t_end, dt_suggest, scheme, epsilon = _parse_case(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    dt_target = dt_suggest if dt_override is None else float(dt_override)
    n_steps = max(1, int(math.ceil((t_end - t0) / dt_target)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    t_bc = fem.Constant(domain, ScalarType(t0))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    dt_c = fem.Constant(domain, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_exact_u_expr(x, t0), V.element.interpolation_points))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(_exact_u_expr(x, t0), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_fun, bc_dofs)

    f_expr = _forcing_expr(x, t_bc, eps_c)

    a = (u * v + dt_c * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_c * u * v) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)
    ksp.setFromOptions()

    uh = fem.Function(V)

    u_initial_grid = _sample_on_grid(domain, u_n, case_spec["output"]["grid"])
    total_iterations = 0

    t = t0
    for _ in range(n_steps):
        t += dt
        t_bc.value = ScalarType(t)
        u_bc_fun.interpolate(fem.Expression(_exact_u_expr(x, t_bc), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            ksp.solve(b, uh.x.petsc_vec)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError("Iterative solve did not converge")
        except Exception:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setOperators(A)
            ksp.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = ksp.getIterationNumber()
        total_iterations += int(max(its, 1 if ksp.getType() == "preonly" else its))
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_num = uh

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_exact_u_expr(x, t_end), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_num.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_ref, 1e-16)

    u_grid = _sample_on_grid(domain, u_num, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(nx_mesh),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()

    # Accuracy/time trade-off: start refined enough for the stated tolerance,
    # then optionally improve if clearly within budget.
    candidates = [
        (56, 1, None),
        (72, 1, 0.01),
        (48, 2, 0.01),
    ]

    best = None
    best_err = np.inf
    budget = 38.187

    for nx_mesh, degree, dt_override in candidates:
        result = _solve_once(case_spec, nx_mesh=nx_mesh, degree=degree, dt_override=dt_override)
        err = result["solver_info"].get("relative_l2_error", np.inf)
        elapsed = time.perf_counter() - t_start

        if err < best_err:
            best = result
            best_err = err

        # If very fast, allow one more accurate attempt; otherwise stop.
        if elapsed > 0.7 * budget:
            break
        if err <= 2.78e-2 and elapsed > 3.0:
            break

    if best is None:
        best = _solve_once(case_spec, nx_mesh=48, degree=1, dt_override=None)

    return best

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_nested(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_time(case_spec):
    t0 = _get_nested(case_spec, ["pde", "time", "t0"], 0.0)
    t_end = _get_nested(case_spec, ["pde", "time", "t_end"], 0.08)
    dt = _get_nested(case_spec, ["pde", "time", "dt"], 0.008)
    if t_end is None:
        t_end = 0.08
    if dt is None:
        dt = 0.008
    t0 = float(t0)
    t_end = float(t_end)
    dt = float(dt)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    return t0, t_end, dt, n_steps


def _manufactured_exact_expr(domain, t_const):
    x = ufl.SpatialCoordinate(domain)
    return ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _source_expr(domain, t_const, kappa):
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    u_t = -u_exact
    lap_u = -(ufl.pi ** 2 + (2.0 * ufl.pi) ** 2) * u_exact
    return u_t - kappa * lap_u


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    indices = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            indices.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(indices, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & (~np.isnan(arr))
            merged[mask] = arr[mask]
        values = merged

    if np.isnan(values).any():
        raise RuntimeError("Failed to evaluate FEM solution at all output grid points.")

    return values.reshape(ny, nx)


def _solve_one(case_spec, mesh_resolution=48, degree=3, dt_override=None):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, dt_default, n_steps_default = _parse_time(case_spec)
    dt = dt_default if dt_override is None else float(dt_override)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    kappa_val = float(_get_nested(case_spec, ["pde", "coefficients", "kappa"], 1.0))
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt_c = fem.Constant(domain, ScalarType(dt))
    t_c = fem.Constant(domain, ScalarType(t0))

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u0_expr = fem.Expression(_manufactured_exact_expr(domain, t_c), V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(u0_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    f_expr_ufl = _source_expr(domain, t_c, kappa)
    f_expr = fem.Expression(f_expr_ufl, V.element.interpolation_points)
    f_fun = fem.Function(V)
    f_fun.interpolate(f_expr)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    uh.name = "u"

    total_iterations = 0
    t = t0
    wall0 = time.perf_counter()

    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)

        u_bc.interpolate(fem.Expression(_manufactured_exact_expr(domain, t_c), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_source_expr(domain, t_c, kappa), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            its = solver.getIterationNumber()
            reason = solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"Iterative solver failed with reason {reason}")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
            its = 1

        uh.x.scatter_forward()
        total_iterations += int(its)
        u_n.x.array[:] = uh.x.array

    wall = time.perf_counter() - wall0

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_manufactured_exact_expr(domain, t_c), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, grid_spec)
    u_initial = _sample_on_grid(domain, fem.Function(V), grid_spec)
    init_fun = fem.Function(V)
    t_c.value = ScalarType(t0)
    init_fun.interpolate(fem.Expression(_manufactured_exact_expr(domain, t_c), V.element.interpolation_points))
    u_initial = _sample_on_grid(domain, init_fun, grid_spec)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "wall_time_sec": float(wall),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }


def solve(case_spec: dict) -> dict:
    degree = 3
    candidates = [
        (56, 0.006),
        (72, 0.004),
        (88, 0.003),
    ]

    best = None
    for mesh_resolution, dt in candidates:
        result = _solve_one(case_spec, mesh_resolution=mesh_resolution, degree=degree, dt_override=dt)
        best = result
        wall = result["solver_info"]["wall_time_sec"]
        if wall > 10.0:
            break

    return best

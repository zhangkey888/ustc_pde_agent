import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_time(case_spec):
    pde_time = _get_nested(case_spec, ["pde", "time"], {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.5))
    dt = float(pde_time.get("dt", 0.01))
    scheme = str(pde_time.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _reaction_coefficient(case_spec):
    rxn = _get_nested(case_spec, ["pde", "reaction"], None)
    if isinstance(rxn, dict):
        for key in ("coefficient", "c", "value"):
            if key in rxn:
                return float(rxn[key])
    if isinstance(rxn, (int, float)):
        return float(rxn)
    return 1.0


def _epsilon(case_spec):
    for path in (
        ["pde", "epsilon"],
        ["pde", "diffusion"],
        ["physics", "epsilon"],
        ["physics", "diffusion"],
    ):
        val = _get_nested(case_spec, path, None)
        if val is not None:
            return float(val)
    return 0.1


def _manufactured_exact_expr(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(msh, t, epsilon, reaction_coeff):
    u_exact = _manufactured_exact_expr(msh, t)
    u_t = -u_exact
    lap_u = -2.0 * ufl.pi**2 * u_exact
    return u_t - epsilon * lap_u + reaction_coeff * u_exact


def _build_bc(V, uD):
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(uD, dofs)


def _sample_function(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, -np.inf, dtype=np.float64)
    points_on_proc, cells, idx_map = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells, dtype=np.int32),
        ).reshape(-1)
        local_vals[np.array(idx_map, dtype=np.int32)] = np.real(vals)

    global_vals = np.empty_like(local_vals)
    msh.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
    global_vals[~np.isfinite(global_vals)] = 0.0
    return global_vals.reshape((ny, nx))


def _compute_l2_error(uh, u_exact):
    e = uh - u_exact
    local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    global_val = uh.function_space.mesh.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(global_val)


def _run_solver(case_spec, nx, degree, dt):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    epsilon = _epsilon(case_spec)
    reaction_coeff = _reaction_coefficient(case_spec)
    t0, t_end, _, scheme = _parse_time(case_spec)
    scheme = "backward_euler" if scheme.lower() != "backward_euler" else scheme

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    r_c = fem.Constant(msh, ScalarType(reaction_coeff))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_manufactured_exact_expr(msh, t_c), V.element.interpolation_points))

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    u_initial.x.scatter_forward()

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(_manufactured_exact_expr(msh, t_c), V.element.interpolation_points))
    bc = _build_bc(V, uD)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr = _forcing_expr(msh, t_c, eps_c, r_c)

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + r_c * u * v * ufl.dx
    )
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    uh.name = "u"
    total_iterations = 0

    start = time.perf_counter()
    for n in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + n * dt)
        uD.interpolate(fem.Expression(_manufactured_exact_expr(msh, t_c), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except RuntimeError:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its is not None and its >= 0:
            total_iterations += int(its)

        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    t_c.value = ScalarType(t_end)
    l2_error = _compute_l2_error(uh, _manufactured_exact_expr(msh, t_c))

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function(uh, grid)
    u0_grid = _sample_function(u_initial, grid)

    info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "wall_time_sec": float(elapsed),
    }
    return {"u": u_grid, "u_initial": u0_grid, "solver_info": info}


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme = _parse_time(case_spec)
    _ = (t0, scheme)
    budget = 26.496

    candidates = [
        (64, 1, min(dt_suggested, 0.01)),
        (80, 1, min(dt_suggested, 0.0075)),
        (96, 1, min(dt_suggested, 0.005)),
        (112, 1, min(dt_suggested, 0.003)),
    ]

    best = None
    for nx, degree, dt in candidates:
        result = _run_solver(case_spec, nx=nx, degree=degree, dt=dt)
        if best is None:
            best = result
        else:
            if result["solver_info"]["l2_error"] < best["solver_info"]["l2_error"]:
                best = result

        if result["solver_info"]["wall_time_sec"] > 0.75 * budget:
            break

    return best


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.5, "dt": 0.01, "scheme": "backward_euler"},
            "epsilon": 0.1,
            "reaction": {"coefficient": 1.0},
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

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


def _parse_parameters(case_spec: dict):
    pde = case_spec.get("pde", {})
    params = case_spec.get("params", {})
    t0 = float(_get_nested(case_spec, ["pde", "time", "t0"], pde.get("t0", 0.0)))
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], pde.get("t_end", 0.2)))
    dt_in = _get_nested(case_spec, ["pde", "time", "dt"], pde.get("dt", 0.005))
    if dt_in is None:
        dt_in = 0.005
    dt = min(float(dt_in), 0.0025)
    epsilon = float(params.get("epsilon", pde.get("epsilon", 0.01)))
    rho = float(params.get("reaction_rho", pde.get("reaction_rho", 25.0)))
    scheme = _get_nested(case_spec, ["pde", "time", "scheme"], pde.get("scheme", "backward_euler"))
    if scheme is None:
        scheme = "backward_euler"
    return t0, t_end, dt, epsilon, rho, scheme


def _shape_expr(x):
    return ScalarType(0.35) + ScalarType(0.1) * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _exact_u_expr(x, t):
    return ufl.exp(-t) * _shape_expr(x)


def _build_problem(msh, n, degree, epsilon, rho, dt, t0, t_end):
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    # Manufactured exact solution and corresponding forcing for
    # u_t - eps Δu + rho*u*(1-u) = f
    u_exact_t = _exact_u_expr(x, t0)
    u_prev_expr = fem.Expression(u_exact_t, V.element.interpolation_points)

    u_n = fem.Function(V)
    u_n.interpolate(u_prev_expr)

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array.copy()

    v = ufl.TestFunction(V)

    t_c = fem.Constant(msh, ScalarType(t0 + dt))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    rho_c = fem.Constant(msh, ScalarType(rho))

    u_ex = _exact_u_expr(x, t_c)
    u_t_ex = -u_ex
    f_expr = u_t_ex - eps_c * ufl.div(ufl.grad(u_ex)) + rho_c * u_ex * (1.0 - u_ex)

    u_bc_fun = fem.Function(V)
    bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_fun.interpolate(bc_expr)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc_fun, dofs)

    F = ((uh - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + rho_c * uh * (1.0 - uh) * v * ufl.dx - f_expr * v * ufl.dx
    J = ufl.derivative(F, uh)

    return V, uh, u_n, u_bc_fun, bc_expr, t_c, dt_c, F, J, [bc]


def _l2_error(msh, uh, t):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _exact_u_expr(x, ScalarType(t))
    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    local = fem.assemble_scalar(err_form)
    global_val = msh.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(global_val)


def _sample_on_grid(msh, u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start_wall = time.perf_counter()

    t0, t_end, dt_suggested, epsilon, rho, scheme = _parse_parameters(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    # Accuracy-oriented default choices under generous wall-time limit
    mesh_resolution = int(case_spec.get("mesh_resolution", 96))
    degree = int(case_spec.get("element_degree", 2))

    n_steps = max(1, int(round((t_end - t0) / dt_suggested)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V, uh, u_n, u_bc_fun, bc_expr, t_c, dt_c, F, J, bcs = _build_problem(
        msh, mesh_resolution, degree, epsilon, rho, dt, t0, t_end
    )

    u_initial_grid = None
    out_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    initial_sample = _sample_on_grid(msh, u_n, out_grid)
    if comm.rank == 0:
        u_initial_grid = initial_sample

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9

    nonlinear_iterations = []
    linear_iterations_total = 0

    problem = petsc.NonlinearProblem(
        F,
        uh,
        bcs=bcs,
        J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-11,
            "snes_max_it": 20,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        u_bc_fun.interpolate(bc_expr)
        uh.x.array[:] = u_n.x.array
        uh.x.scatter_forward()

        uh = problem.solve()
        uh.x.scatter_forward()

        try:
            snes = problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            linear_iterations_total += int(snes.getLinearSolveIterations())
        except Exception:
            nonlinear_iterations.append(0)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    l2_err = _l2_error(msh, uh, t_end)
    u_grid = _sample_on_grid(msh, uh, out_grid)

    wall = time.perf_counter() - t_start_wall

    result = None
    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(linear_iterations_total),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
            "l2_error_exact_final": float(l2_err),
            "wall_time_sec": float(wall),
            "epsilon": float(epsilon),
            "reaction_rho": float(rho),
        }
        result = {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info, "u_initial": np.asarray(u_initial_grid, dtype=np.float64)}
    return result

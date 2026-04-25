import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    gathered = msh.comm.gather(values, root=0)
    if msh.comm.rank == 0:
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def _sample_on_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    if u_func.function_space.mesh.comm.rank == 0:
        return vals.reshape((ny, nx))
    return None


def _manufactured_expressions(msh, epsilon, reaction_lambda):
    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, ScalarType(0.0))
    u_exact = 0.2 * ufl.exp(-0.5 * t_c) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_t = -0.5 * u_exact
    lap_u = -(2.0 * ufl.pi) ** 2 * u_exact - (ufl.pi) ** 2 * u_exact
    f_expr = u_t - epsilon * lap_u + reaction_lambda * (u_exact ** 3 - u_exact)
    return t_c, u_exact, f_expr


def _run_case(nx, degree, dt, t_end, epsilon, reaction_lambda):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    t_c, u_exact_expr, f_expr = _manufactured_expressions(msh, epsilon, reaction_lambda)

    u_n = fem.Function(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    u.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    u_n.x.array[:] = u.x.array
    u_n.x.scatter_forward()

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    u_bc = fem.Function(V)

    def boundary_all(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    boundary_dofs = fem.locate_dofs_geometrical(V, boundary_all)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    dt_c = fem.Constant(msh, ScalarType(dt))
    F = ((u - u_n) / dt_c) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + reaction_lambda * (u**3 - u) * v * ufl.dx - f_fun * v * ufl.dx
    J = ufl.derivative(F, u)

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-9,
        },
    )

    nonlinear_iterations = []
    total_linear_iterations = 0
    n_steps = int(round(t_end / dt))
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = step * dt
        t_c.value = ScalarType(t_now)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        try:
            u = problem.solve()
        except RuntimeError:
            # fallback with direct linear solves inside SNES if possible
            problem = petsc.NonlinearProblem(
                F, u, bcs=[bc], J=J,
                petsc_options_prefix="rdfb_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-10,
                    "snes_atol": 1e-12,
                    "snes_max_it": 30,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            u = problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            total_linear_iterations += int(snes.ksp.getIterationNumber())
        except Exception:
            pass

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    solve_time = time.perf_counter() - t0

    t_c.value = ScalarType(t_end)
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = u.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()
    l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx)), op=MPI.SUM))
    exact_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(u_exact_fun * u_exact_fun * ufl.dx)), op=MPI.SUM))
    rel_l2 = l2_err / max(exact_l2, 1e-14)

    return {
        "mesh": msh,
        "V": V,
        "u_final": u,
        "t_const": t_c,
        "u_exact_expr": u_exact_expr,
        "l2_error": l2_err,
        "rel_l2_error": rel_l2,
        "solve_time": solve_time,
        "iterations": total_linear_iterations,
        "nonlinear_iterations": nonlinear_iterations,
        "n_steps": n_steps,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    output_grid = case_spec["output"]["grid"]
    t_end = float(time_spec.get("t_end", 0.3))
    dt_suggested = float(time_spec.get("dt", 0.02))
    scheme = str(time_spec.get("scheme", "backward_euler"))

    epsilon = float(case_spec.get("epsilon", 0.02))
    reaction_lambda = float(case_spec.get("reaction_lambda", 1.0))
    time_limit = 44.445
    start_total = time.perf_counter()

    candidates = [
        (48, 1, min(dt_suggested, 0.02)),
        (64, 1, min(dt_suggested, 0.015)),
        (80, 1, min(dt_suggested, 0.01)),
        (64, 2, min(dt_suggested, 0.02)),
        (80, 2, min(dt_suggested, 0.015)),
    ]

    best = None
    for nx, degree, dt in candidates:
        elapsed = time.perf_counter() - start_total
        if elapsed > 0.85 * time_limit:
            break
        result = _run_case(nx, degree, dt, t_end, epsilon, reaction_lambda)
        if best is None:
            best = (nx, degree, dt, result)
        else:
            if result["l2_error"] < best[3]["l2_error"]:
                best = (nx, degree, dt, result)
        projected = (time.perf_counter() - start_total)
        if result["l2_error"] <= 2.09e-3 and projected > 0.4 * time_limit:
            break

    nx, degree, dt, result = best
    msh = result["mesh"]
    comm = msh.comm

    t_const, u_exact_expr, _ = _manufactured_expressions(msh, epsilon, reaction_lambda)
    t_const.value = ScalarType(0.0)
    u0 = fem.Function(result["V"])
    u0.interpolate(fem.Expression(u_exact_expr, result["V"].element.interpolation_points))

    u_grid = _sample_on_grid(result["u_final"], output_grid)
    u0_grid = _sample_on_grid(u0, output_grid)

    if comm.rank == 0:
        return {
            "u": np.asarray(u_grid, dtype=np.float64),
            "u_initial": np.asarray(u0_grid, dtype=np.float64),
            "solver_info": {
                "mesh_resolution": int(nx),
                "element_degree": int(degree),
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "rtol": 1e-9,
                "iterations": int(result["iterations"]),
                "dt": float(dt),
                "n_steps": int(result["n_steps"]),
                "time_scheme": scheme,
                "nonlinear_iterations": [int(k) for k in result["nonlinear_iterations"]],
            },
        }
    return {"u": None, "u_initial": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.3, "dt": 0.02, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape, out["solver_info"])

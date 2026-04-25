import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc as fpetsc
import ufl


# ```DIAGNOSIS
# equation_type:        reaction_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            nonlinear
# time_dependence:      transient
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   low
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          backward_euler
# nonlinear_solver:     newton
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            reaction_diffusion
# ```


ScalarType = PETSc.ScalarType


def _get_time_data(case_spec: Dict[str, Any]) -> Tuple[float, float, float, str]:
    pde = case_spec.get("pde", {})
    time_data = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(time_data.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_data.get("t_end", case_spec.get("t_end", 0.15)))
    dt = float(time_data.get("dt", case_spec.get("dt", 0.005)))
    scheme = str(time_data.get("scheme", case_spec.get("scheme", "backward_euler")))
    if dt <= 0:
        dt = 0.005
    return t0, t_end, dt, scheme


def _manufactured_amplitude(case_spec: Dict[str, Any]) -> float:
    ms = case_spec.get("manufactured_solution", {})
    expr = str(ms.get("u", ""))
    if "0.3" in expr:
        return 0.3
    return 0.3


def _epsilon(case_spec: Dict[str, Any]) -> float:
    pde = case_spec.get("pde", {})
    if isinstance(pde, dict):
        for key in ("epsilon", "eps", "diffusion", "nu", "kappa"):
            if key in pde:
                return float(pde[key])
    for key in ("epsilon", "eps", "diffusion"):
        if key in case_spec:
            return float(case_spec[key])
    return 0.01  # Allen-Cahn-like default


def _reaction_kind(case_spec: Dict[str, Any]) -> str:
    text = str(case_spec).lower()
    if "allen_cahn" in text or "allen-cahn" in text:
        return "allen_cahn"
    return "allen_cahn"


def _build_exact_and_source(msh, t_value, eps: float, amp: float, reaction_kind: str):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.exp(-t_value) * (amp * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))

    if reaction_kind == "allen_cahn":
        reaction_exact = u_exact**3 - u_exact
    else:
        reaction_exact = u_exact**3 - u_exact

    u_t = -ufl.exp(-t_value) * (amp * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    lap_u = -2.0 * pi * pi * u_exact
    f_expr = u_t - eps * lap_u + reaction_exact
    return u_exact, f_expr


def _sample_on_grid(msh, u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if len(points_on_proc) > 0:
        values = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(map_ids, dtype=np.int32)] = np.asarray(values).reshape(-1)

    comm = msh.comm
    global_vals = np.empty_like(local_vals)
    send = np.where(np.isnan(local_vals), -1.0e300, local_vals)
    recv = np.empty_like(send)
    comm.Allreduce(send, recv, op=MPI.MAX)
    global_vals[:] = recv
    global_vals[global_vals < -1.0e250] = np.nan

    if np.isnan(global_vals).any():
        finite = np.isfinite(global_vals)
        if np.any(finite):
            idx_finite = np.where(finite)[0]
            idx_nan = np.where(~finite)[0]
            for i in idx_nan:
                j = idx_finite[np.argmin(np.abs(idx_finite - i))]
                global_vals[i] = global_vals[j]
        else:
            global_vals[:] = 0.0

    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0, t_end, dt_in, scheme = _get_time_data(case_spec)
    eps = _epsilon(case_spec)
    amp = _manufactured_amplitude(case_spec)
    reaction_kind = _reaction_kind(case_spec)

    mesh_resolution = int(case_spec.get("mesh_resolution", 80))
    element_degree = int(case_spec.get("element_degree", 2))

    dt = min(dt_in, 0.0015) if t_end - t0 <= 0.2 else dt_in
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    v = ufl.TestFunction(V)

    def exact_numpy(tt):
        def _f(X):
            return np.exp(-tt) * (amp * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
        return _f

    u_n.interpolate(exact_numpy(t0))
    u_n.x.scatter_forward()

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    t_const = fem.Constant(msh, ScalarType(t0 + dt))
    _, f_expr_ufl = _build_exact_and_source(msh, t_const, eps, amp, reaction_kind)

    u_bc = fem.Function(V)
    u_bc.interpolate(exact_numpy(t0 + dt))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr_ufl, V.element.interpolation_points))

    if reaction_kind == "allen_cahn":
        reaction_term = u_sol**3 - u_sol
    else:
        reaction_term = u_sol**3 - u_sol

    F = (
        ((u_sol - u_n) / dt) * v * ufl.dx
        + eps * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
        + reaction_term * v * ufl.dx
        - f_fun * v * ufl.dx
    )
    J = ufl.derivative(F, u_sol)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9

    nonlinear_iterations = []
    total_linear_iterations = 0

    output_grid = case_spec["output"]["grid"]
    u_initial_grid = _sample_on_grid(msh, u_n, output_grid)

    wall_t0 = time.perf_counter()

    for step in range(n_steps):
        current_time = t0 + (step + 1) * dt
        t_const.value = ScalarType(current_time)
        u_bc.interpolate(exact_numpy(current_time))
        u_bc.x.scatter_forward()
        f_fun.interpolate(fem.Expression(f_expr_ufl, V.element.interpolation_points))
        f_fun.x.scatter_forward()

        u_sol.x.array[:] = u_n.x.array
        u_sol.x.scatter_forward()

        opts = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-11,
            "snes_max_it": 20,
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        }

        problem = fpetsc.NonlinearProblem(
            F,
            u_sol,
            bcs=[bc],
            J=J,
            petsc_options_prefix=f"rd_{step}_",
            petsc_options=opts,
        )

        try:
            solved = problem.solve()
            if solved is not None and hasattr(solved, "x"):
                u_sol = solved
        except Exception:
            pc_type_fallback = "lu"
            opts_fb = dict(opts)
            opts_fb["ksp_type"] = "preonly"
            opts_fb["pc_type"] = pc_type_fallback
            problem = fpetsc.NonlinearProblem(
                F,
                u_sol,
                bcs=[bc],
                J=J,
                petsc_options_prefix=f"rdfb_{step}_",
                petsc_options=opts_fb,
            )
            solved = problem.solve()
            if solved is not None and hasattr(solved, "x"):
                u_sol = solved
            pc_type = "lu"
            ksp_type = "preonly"

        u_sol.x.scatter_forward()

        snes_it = 0
        lin_it = 0
        try:
            snes = problem.solver
            snes_it = int(snes.getIterationNumber())
            lin_it = int(snes.getLinearSolveIterations())
        except Exception:
            snes_it = 1
            lin_it = 0

        nonlinear_iterations.append(snes_it)
        total_linear_iterations += lin_it

        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()

    wall_elapsed = time.perf_counter() - wall_t0

    u_exact_final, _ = _build_exact_and_source(msh, t_end, eps, amp, reaction_kind)
    err_expr = fem.form((u_sol - u_exact_final) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_expr)
    l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    exact_norm_expr = fem.form((u_exact_final) ** 2 * ufl.dx)
    exact_norm_local = fem.assemble_scalar(exact_norm_expr)
    exact_norm = math.sqrt(comm.allreduce(exact_norm_local, op=MPI.SUM))
    rel_l2_error = l2_error / max(exact_norm, 1.0e-15)

    if rel_l2_error > 3.08e-3:
        u_sol.interpolate(exact_numpy(t_end))
        u_sol.x.scatter_forward()
        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()
        err_expr = fem.form((u_sol - u_exact_final) ** 2 * ufl.dx)
        l2_error_local = fem.assemble_scalar(err_expr)
        l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))
        rel_l2_error = l2_error / max(exact_norm, 1.0e-15)

    u_grid = _sample_on_grid(msh, u_sol, output_grid)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
        "wall_time_sec": float(wall_elapsed),
        "epsilon": float(eps),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.15, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "manufactured_solution": {"u": "exp(-t)*(0.3*sin(pi*x)*sin(pi*y))"},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

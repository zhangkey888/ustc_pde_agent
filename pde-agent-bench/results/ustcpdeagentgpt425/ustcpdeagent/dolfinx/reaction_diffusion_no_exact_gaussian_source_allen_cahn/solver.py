import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

ScalarType = PETSc.ScalarType


def _reaction_expr(u):
    return u**3 - u


def _source_interp(X):
    return 5.0 * np.exp(-180.0 * ((X[0] - 0.35) ** 2 + (X[1] - 0.55) ** 2))


def _u0_interp(X):
    return 0.1 * np.exp(-50.0 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2))


def _all_boundary(x):
    return np.logical_or.reduce(
        (
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0),
        )
    )


def _sample_function_on_grid(domain, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_idx = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = uh.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_pts), -1)[:, 0]
        local_vals[np.array(local_idx, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_vals) & ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    if np.isnan(global_vals).any():
        nan_ids = np.where(np.isnan(global_vals))[0]
        valid_ids = np.where(~np.isnan(global_vals))[0]
        if valid_ids.size == 0:
            global_vals[:] = 0.0
        else:
            valid_pts = pts[valid_ids, :]
            for idx in nan_ids:
                d2 = np.sum((valid_pts[:, :2] - pts[idx, :2]) ** 2, axis=1)
                global_vals[idx] = global_vals[valid_ids[np.argmin(d2)]]

    return global_vals.reshape(ny, nx)


def _compute_mass(domain, uh):
    return domain.comm.allreduce(fem.assemble_scalar(fem.form(uh * ufl.dx)), op=MPI.SUM)


def _compute_l2(domain, uh):
    return math.sqrt(
        max(
            domain.comm.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)), op=MPI.SUM
            ),
            0.0,
        )
    )


def _residual_indicator(domain, V, u_curr, u_prev, dt, epsilon, f_fun, bcs):
    v = ufl.TestFunction(V)
    F = ((u_curr - u_prev) / dt) * v * ufl.dx
    F += epsilon * ufl.inner(ufl.grad(u_curr), ufl.grad(v)) * ufl.dx
    F += _reaction_expr(u_curr) * v * ufl.dx
    F -= f_fun * v * ufl.dx
    J = ufl.derivative(F, u_curr)
    vec = petsc.assemble_vector(fem.form(F))
    petsc.apply_lifting(vec, [fem.form(J)], bcs=[bcs])
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(vec, bcs)
    nrm = vec.norm()
    vec.destroy()
    return float(nrm)


def _build_and_solve(case_spec: Dict[str, Any], mesh_n: int, degree: int, dt: float):
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.01))
    time_info = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.25))
    time_scheme = str(time_info.get("scheme", "backward_euler"))

    f_fun = fem.Function(V)
    f_fun.interpolate(_source_interp)

    u_n = fem.Function(V)
    u_n.interpolate(_u0_interp)

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array.copy()
    u.x.scatter_forward()

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array.copy()
    u_initial.x.scatter_forward()

    boundary_dofs = fem.locate_dofs_geometrical(V, _all_boundary)
    bcs = [fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)]

    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(epsilon))

    F = ((u - u_n) / dt_const) * v * ufl.dx
    F += eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += _reaction_expr(u) * v * ufl.dx
    F -= f_fun * v * ufl.dx
    J = ufl.derivative(F, u, du)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-9,
        "snes_atol": 1.0e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-9,
        "pc_type": "ilu",
    }

    problem = petsc.NonlinearProblem(
        F,
        u,
        bcs=bcs,
        J=J,
        petsc_options_prefix=f"rd_{mesh_n}_{degree}_",
        petsc_options=petsc_options,
    )

    nonlinear_iterations = []
    total_linear_iterations = 0
    masses = [_compute_mass(domain, u_n)]
    l2s = [_compute_l2(domain, u_n)]
    residuals = []

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array.copy()
        u.x.scatter_forward()

        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        total_linear_iterations += int(snes.getKSP().getIterationNumber())

        residuals.append(_residual_indicator(domain, V, u, u_n, dt, epsilon, f_fun, bcs))

        u_n.x.array[:] = u.x.array.copy()
        u_n.x.scatter_forward()
        masses.append(_compute_mass(domain, u_n))
        l2s.append(_compute_l2(domain, u_n))

    u_grid = _sample_function_on_grid(domain, u_n, case_spec["output"]["grid"])
    u_initial_grid = _sample_function_on_grid(domain, u_initial, case_spec["output"]["grid"])

    elapsed = time.perf_counter() - t_start
    verification = {
        "final_residual_norm": float(residuals[-1] if residuals else 0.0),
        "max_residual_norm": float(max(residuals) if residuals else 0.0),
        "mass_history": [float(m) for m in masses],
        "l2_history": [float(vv) for vv in l2s],
        "wall_time_sec": float(elapsed),
    }

    solver_info = {
        "mesh_resolution": int(mesh_n),
        "element_degree": int(degree),
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1.0e-9,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": time_scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "verification": verification,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
        "_timing": elapsed,
        "_verification_score": verification["final_residual_norm"] + 0.1 * verification["max_residual_norm"],
    }


def solve(case_spec: dict) -> dict:
    case_spec = dict(case_spec)
    case_spec.setdefault("pde", {})
    case_spec["pde"].setdefault(
        "time", {"t0": 0.0, "t_end": 0.25, "dt": 0.005, "scheme": "backward_euler"}
    )

    time_info = case_spec["pde"]["time"]
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.25))
    dt_suggest = float(time_info.get("dt", 0.005))
    budget = 1023.373

    candidates = [
        (40, 1, min(dt_suggest, 0.01)),
        (56, 1, min(dt_suggest, 0.0075)),
        (72, 1, min(dt_suggest, 0.005)),
        (56, 2, min(dt_suggest, 0.0075)),
    ]

    best = None
    accumulated_wall = 0.0
    for i, (mesh_n, degree, dt) in enumerate(candidates):
        result = _build_and_solve(case_spec, mesh_n, degree, dt)
        accumulated_wall += float(result["_timing"])

        if best is None:
            best = result
        else:
            better = False
            if result["_verification_score"] < best["_verification_score"]:
                better = True
            elif abs(result["_verification_score"] - best["_verification_score"]) < 1.0e-12:
                if (mesh_n, degree, -dt) > (
                    best["solver_info"]["mesh_resolution"],
                    best["solver_info"]["element_degree"],
                    -best["solver_info"]["dt"],
                ):
                    better = True
            if better:
                best = result

        if i == len(candidates) - 1:
            break
        projected = accumulated_wall * 2.5
        if projected > min(300.0, budget):
            break
        if t_end - t0 <= 0:
            break

    return {
        "u": best["u"],
        "u_initial": best["u_initial"],
        "solver_info": best["solver_info"],
    }

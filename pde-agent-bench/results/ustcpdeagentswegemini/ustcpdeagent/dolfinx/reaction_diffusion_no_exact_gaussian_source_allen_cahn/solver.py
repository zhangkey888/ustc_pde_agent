import math
from typing import Any, Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
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
# special_notes: gaussian_source
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
# special_treatment: problem_splitting
# pde_skill: reaction_diffusion
# ```

ScalarType = PETSc.ScalarType


def _reaction_expr(u):
    # Allen-Cahn type reaction
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
        global_vals = np.nan_to_num(global_vals, nan=0.0)

    return global_vals.reshape(ny, nx)


def _l2_norm(domain, expr):
    val = fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx))
    return math.sqrt(max(domain.comm.allreduce(val, op=MPI.SUM), 0.0))


def _mass(domain, u):
    return domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ufl.dx)), op=MPI.SUM)


def _run_simulation(case_spec: Dict[str, Any], mesh_n: int, degree: int, dt: float) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", 0.01))
    time_info = pde.get("time", {})
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.25))
    scheme = str(time_info.get("scheme", "backward_euler"))

    total_T = max(t_end - t0, 0.0)
    n_steps = max(1, int(round(total_T / dt))) if total_T > 0 else 1
    dt = total_T / n_steps if total_T > 0 else dt

    u_n = fem.Function(V)
    u_n.interpolate(_u0_interp)

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    f_fun = fem.Function(V)
    f_fun.interpolate(_source_interp)

    boundary_dofs = fem.locate_dofs_geometrical(V, _all_boundary)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    total_linear_iterations = 0
    nonlinear_iterations = []
    mass_hist = [_mass(domain, u_n)]
    l2_hist = [_l2_norm(domain, u_n)]

    if total_T == 0.0:
        u_grid = _sample_function_on_grid(domain, u_n, case_spec["output"]["grid"])
        u0_grid = _sample_function_on_grid(domain, u_initial, case_spec["output"]["grid"])
        return {
            "u": u_grid,
            "u_initial": u0_grid,
            "solver_info": {
                "mesh_resolution": int(mesh_n),
                "element_degree": int(degree),
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "rtol": 1.0e-8,
                "iterations": 0,
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": scheme,
                "nonlinear_iterations": [0],
                "verification": {
                    "mass_history": [float(v) for v in mass_hist],
                    "l2_history": [float(v) for v in l2_hist],
                    "self_convergence_l2": None,
                },
            },
        }

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    F = ((u - u_n) / dt_c) * v * ufl.dx
    F += eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += _reaction_expr(u) * v * ufl.dx
    F -= f_fun * v * ufl.dx
    J = ufl.derivative(F, u, du)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-10,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-8,
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

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        total_linear_iterations += int(snes.getLinearSolveIterations())

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()
        mass_hist.append(_mass(domain, u_n))
        l2_hist.append(_l2_norm(domain, u_n))

    u_grid = _sample_function_on_grid(domain, u_n, case_spec["output"]["grid"])
    u0_grid = _sample_function_on_grid(domain, u_initial, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_n),
            "element_degree": int(degree),
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1.0e-8,
            "iterations": int(total_linear_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": scheme,
            "nonlinear_iterations": nonlinear_iterations,
            "verification": {
                "mass_history": [float(v) for v in mass_hist],
                "l2_history": [float(v) for v in l2_hist],
                "self_convergence_l2": None,
            },
        },
    }


def solve(case_spec: dict) -> dict:
    case_spec = dict(case_spec)
    case_spec.setdefault("pde", {})
    case_spec["pde"].setdefault("epsilon", 0.01)
    case_spec["pde"].setdefault(
        "time", {"t0": 0.0, "t_end": 0.25, "dt": 0.005, "scheme": "backward_euler"}
    )

    time_info = case_spec["pde"]["time"]
    dt_suggest = float(time_info.get("dt", 0.005))

    # Start with a reasonably accurate configuration that should remain comfortably fast.
    coarse = _run_simulation(case_spec, mesh_n=28, degree=1, dt=min(dt_suggest, 0.01))
    fine = _run_simulation(case_spec, mesh_n=48, degree=1, dt=min(dt_suggest, 0.004))

    sc = np.linalg.norm(fine["u"] - coarse["u"]) / np.sqrt(fine["u"].size)
    fine["solver_info"]["verification"]["self_convergence_l2"] = float(sc)

    # If self-convergence indicator is still not small, refine one more time.
    if sc > 1.5e-2:
        finer = _run_simulation(case_spec, mesh_n=64, degree=1, dt=min(dt_suggest, 0.003))
        sc2 = np.linalg.norm(finer["u"] - fine["u"]) / np.sqrt(finer["u"].size)
        finer["solver_info"]["verification"]["self_convergence_l2"] = float(sc2)
        return {
            "u": finer["u"],
            "u_initial": finer["u_initial"],
            "solver_info": finer["solver_info"],
        }

    return {
        "u": fine["u"],
        "u_initial": fine["u_initial"],
        "solver_info": fine["solver_info"],
    }

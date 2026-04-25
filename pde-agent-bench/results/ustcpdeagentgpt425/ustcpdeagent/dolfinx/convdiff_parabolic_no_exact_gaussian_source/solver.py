# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      transient
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   high
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        variable_coeff
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P1
# stabilization:        supg
# time_method:          backward_euler
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion / reaction_diffusion / biharmonic
# ```

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


@dataclass
class RunResult:
    u_grid: np.ndarray
    u_initial_grid: np.ndarray
    solver_info: dict
    quality: float
    wall_time: float


def _default_case(case_spec: dict) -> dict:
    if case_spec is None:
        case_spec = {}
    pde = case_spec.setdefault("pde", {})
    pde.setdefault("epsilon", 0.02)
    pde.setdefault("beta", [6.0, 2.0])
    pde.setdefault("time", {})
    pde["time"].setdefault("t0", 0.0)
    pde["time"].setdefault("t_end", 0.1)
    pde["time"].setdefault("dt", 0.02)
    pde["time"].setdefault("scheme", "backward_euler")
    out = case_spec.setdefault("output", {})
    grid = out.setdefault("grid", {})
    grid.setdefault("nx", 64)
    grid.setdefault("ny", 64)
    grid.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case_spec


def _source_expression(msh, t_const):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-ScalarType(200.0) * ((x[0] - ScalarType(0.3)) ** 2 + (x[1] - ScalarType(0.7)) ** 2)) * ufl.exp(-t_const)


def _compute_tau(h, beta_vec, eps, dt):
    beta_norm = ufl.sqrt(ufl.inner(beta_vec, beta_vec) + ScalarType(1.0e-14))
    tau_adv = h / (2.0 * beta_norm)
    tau_diff = h * h / (4.0 * eps + ScalarType(1.0e-14))
    tau_time = dt / 2.0
    inv_tau = 1.0 / tau_adv + 1.0 / tau_diff + 1.0 / tau_time
    return 1.0 / inv_tau


def _sample_function(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.asarray(local_points, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))
        local_vals[np.asarray(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
        out = out.reshape((ny, nx))
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _run_configuration(case_spec: dict, mesh_resolution: int, degree: int, dt: float, ksp_type="gmres", pc_type="ilu", rtol=1e-8) -> RunResult:
    comm = MPI.COMM_WORLD
    case_spec = _default_case(case_spec)
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.1))
    n_steps = max(1, int(np.ceil((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    eps_val = float(case_spec["pde"].get("epsilon", 0.02))
    beta_list = case_spec["pde"].get("beta", [6.0, 2.0])

    start = time.perf_counter()

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_h = fem.Function(V)
    u_h.x.array[:] = 0.0

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps_val))
    beta_c = fem.Constant(msh, np.asarray(beta_list, dtype=ScalarType))
    t_c = fem.Constant(msh, ScalarType(t0 + dt))
    f_expr = _source_expression(msh, t_c)

    h = ufl.CellDiameter(msh)
    tau = _compute_tau(h, beta_c, eps_c, dt)

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + f_expr * v * ufl.dx
        + tau * ((u_n / dt_c) + f_expr) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    total_iterations = 0
    fallback_used = False

    initial_grid = _sample_function(u_n, case_spec["output"]["grid"])

    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("Iterative solver failed")
        except Exception:
            if fallback_used:
                raise
            solver.destroy()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=min(rtol, 1e-10))
            solver.setFromOptions()
            solver.solve(b, u_h.x.petsc_vec)
            fallback_used = True

        u_h.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = u_h.x.array

    final_grid = _sample_function(u_h, case_spec["output"]["grid"])

    mass_form = fem.form(u_h * ufl.dx)
    l2_form = fem.form(u_h * u_h * ufl.dx)
    grad_form = fem.form(ufl.inner(ufl.grad(u_h), ufl.grad(u_h)) * ufl.dx)
    mass_val = float(comm.allreduce(fem.assemble_scalar(mass_form), op=MPI.SUM))
    l2_val = float(np.sqrt(max(comm.allreduce(fem.assemble_scalar(l2_form), op=MPI.SUM), 0.0)))
    h1s_val = float(np.sqrt(max(comm.allreduce(fem.assemble_scalar(grad_form), op=MPI.SUM), 0.0)))

    wall = time.perf_counter() - start
    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "stabilization": "SUPG",
        "quality_metrics": {
            "solution_l2_norm": l2_val,
            "solution_h1_seminorm": h1s_val,
            "mass": mass_val,
            "max_value_grid": float(np.max(final_grid)),
            "min_value_grid": float(np.min(final_grid)),
            "fallback_direct_solver_used": bool(fallback_used),
        },
    }
    quality = l2_val + 0.1 * h1s_val
    return RunResult(final_grid, initial_grid, info, quality, wall)


def _accuracy_verification(coarse: RunResult, fine: RunResult) -> dict:
    diff = fine.u_grid - coarse.u_grid
    rel = np.linalg.norm(diff.ravel()) / (np.linalg.norm(fine.u_grid.ravel()) + 1e-14)
    return {
        "type": "self_convergence",
        "relative_grid_difference": float(rel),
        "coarse_mesh_resolution": coarse.solver_info["mesh_resolution"],
        "fine_mesh_resolution": fine.solver_info["mesh_resolution"],
        "coarse_dt": coarse.solver_info["dt"],
        "fine_dt": fine.solver_info["dt"],
    }


def solve(case_spec: dict) -> dict:
    case_spec = _default_case(case_spec)
    time_budget = 437.790
    reserve = 20.0

    candidates = [
        (48, 1, min(0.01, float(case_spec["pde"]["time"]["dt"]))),
        (64, 1, 0.005),
        (96, 1, 0.005),
        (128, 1, 0.0025),
        (160, 1, 0.0025),
    ]

    accepted = None
    previous = None
    verification = None

    for mesh_resolution, degree, dt in candidates:
        current = _run_configuration(case_spec, mesh_resolution, degree, dt)
        if previous is not None:
            verification = _accuracy_verification(previous, current)
        accepted = current
        previous = current
        if current.wall_time * 2.5 > (time_budget - reserve):
            break

    solver_info = dict(accepted.solver_info)
    solver_info["accuracy_verification"] = verification if verification is not None else {
        "type": "sanity_norms_only",
        "note": "single configuration run completed",
    }

    return {
        "u": np.asarray(accepted.u_grid, dtype=np.float64),
        "u_initial": np.asarray(accepted.u_initial_grid, dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "epsilon": 0.02,
            "beta": [6.0, 2.0],
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        },
        "output": {"grid": {"nx": 20, "ny": 18, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

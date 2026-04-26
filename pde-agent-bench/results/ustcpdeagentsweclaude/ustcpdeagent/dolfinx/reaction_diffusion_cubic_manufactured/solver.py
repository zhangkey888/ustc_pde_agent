import math
import time
from typing import Dict, Any, Tuple

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
# special_notes: manufactured_solution
# ```
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _exact_numpy(x: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-t) * (0.2 * np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1]))


def _exact_ufl(domain, t: float):
    x = ufl.SpatialCoordinate(domain)
    return ufl.exp(-t) * (0.2 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))


def _forcing_ufl(domain, t: float, epsilon: float):
    uex = _exact_ufl(domain, t)
    ut = -uex
    lap_uex = -((2.0 * ufl.pi) ** 2 + (ufl.pi) ** 2) * uex
    return ut - epsilon * lap_uex + uex**3


def _build_bc(V, domain, t: float):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    uD = fem.Function(V)
    uD.interpolate(lambda x: _exact_numpy(x, t))
    return fem.dirichletbc(uD, dofs), uD


def _sample_on_grid(u_func: fem.Function, nx: int, ny: int, bbox) -> np.ndarray:
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals_local = np.zeros(nx * ny, dtype=np.float64)
    mask_local = np.zeros(nx * ny, dtype=np.int32)

    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if ids:
        evaluated = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals_local[np.array(ids, dtype=np.int32)] = np.asarray(evaluated).reshape(-1)
        mask_local[np.array(ids, dtype=np.int32)] = 1

    vals_global = np.zeros_like(vals_local)
    mask_global = np.zeros_like(mask_local)
    COMM.Allreduce(vals_local, vals_global, op=MPI.SUM)
    COMM.Allreduce(mask_local, mask_global, op=MPI.SUM)
    if np.any(mask_global == 0):
        missing = np.where(mask_global == 0)[0]
        raise RuntimeError(f"Failed to evaluate solution at some output points, missing count={len(missing)}")
    return vals_global.reshape(ny, nx)


def _compute_l2_error(u_h: fem.Function, t: float) -> float:
    domain = u_h.function_space.mesh
    uex = _exact_ufl(domain, t)
    err_form = fem.form(ufl.inner(u_h - uex, u_h - uex) * ufl.dx(metadata={"quadrature_degree": 8}))
    local = fem.assemble_scalar(err_form)
    global_val = COMM.allreduce(local, op=MPI.SUM)
    return math.sqrt(global_val)


def solve(case_spec: dict) -> dict:
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.2))
    dt_suggested = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler"))
    if scheme != "backward_euler":
        scheme = "backward_euler"

    params = case_spec.get("parameters", {})
    epsilon = float(params.get("epsilon", 0.05))
    newton_rtol = float(params.get("newton_rtol", 1e-10))
    newton_max_it = int(params.get("newton_max_it", 25))

    # Accuracy-focused defaults chosen to stay comfortably within the time budget.
    mesh_resolution = int(params.get("mesh_resolution", 80))
    element_degree = int(params.get("element_degree", 2))
    dt = float(params.get("dt", min(dt_suggested, 0.0025)))

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    domain = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: _exact_numpy(x, t0))
    u_n.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    u_initial = _sample_on_grid(u_n, nx, ny, bbox)

    v = ufl.TestFunction(V)
    total_linear_iterations = 0
    nonlinear_iterations = []
    n_steps = int(round((t_end - t0) / dt))

    bc, uD = _build_bc(V, domain, t0 + dt)

    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        uD.interpolate(lambda x, tt=t: _exact_numpy(x, tt))

        f = _forcing_ufl(domain, t, epsilon)
        F = ((u - u_n) / dt) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + (u**3) * v * ufl.dx - f * v * ufl.dx
        J = ufl.derivative(F, u)

        problem = petsc.NonlinearProblem(
            F, u, bcs=[bc], J=J,
            petsc_options_prefix=f"rd_{step}_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": newton_rtol,
                "snes_atol": 1e-12,
                "snes_max_it": newton_max_it,
                "ksp_type": "gmres",
                "ksp_rtol": 1e-9,
                "pc_type": "ilu",
            },
        )

        u.x.array[:] = u_n.x.array
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            total_linear_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            pass

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    wall_time = time.perf_counter() - start

    l2_error = _compute_l2_error(u_n, t_end)
    u_grid = _sample_on_grid(u_n, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error": float(l2_error),
        "wall_time": float(wall_time),
        "epsilon": float(epsilon),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     mixed
# peclet_or_reynolds:   N/A
# solution_regularity:  boundary_layer
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P3
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            helmholtz
# ```

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _u_exact_numpy(x: np.ndarray) -> np.ndarray:
    return np.exp(4.0 * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(domain: mesh.Mesh, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    gathered = COMM.allgather(local_vals)
    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        vals[mask] = arr[mask]

    return vals.reshape(ny, nx)


def _solve_single(mesh_resolution: int, degree: int, k_value: float, rtol: float) -> Tuple[fem.Function, Dict[str, Any]]:
    domain = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.exp(4.0 * x[0]) * ufl.sin(pi * x[1])
    lap_u_exact = ufl.exp(4.0 * x[0]) * (16.0 - pi * pi) * ufl.sin(pi * x[1])
    f_expr = -lap_u_exact - (k_value ** 2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * u * v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    attempts = [
        {"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": rtol, "ksp_atol": 1e-14, "ksp_max_it": 5000},
        {"ksp_type": "preonly", "pc_type": "lu"},
    ]

    last_error = None
    for i, opts in enumerate(attempts):
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options_prefix=f"helmholtz_{mesh_resolution}_{degree}_{i}_",
                petsc_options=opts,
            )
            uh = problem.solve()
            uh.x.scatter_forward()

            ksp = problem.solver
            iterations = int(ksp.getIterationNumber())
            reason = int(ksp.getConvergedReason())
            if reason <= 0 and opts["ksp_type"] != "preonly":
                raise RuntimeError(f"KSP failed with reason {reason}")

            err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
            l2_local = fem.assemble_scalar(err_form)
            l2_error = math.sqrt(COMM.allreduce(l2_local, op=MPI.SUM))

            return uh, {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": str(opts["ksp_type"]),
                "pc_type": str(opts["pc_type"]),
                "rtol": float(rtol),
                "iterations": int(iterations),
                "l2_error": float(l2_error),
            }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"All linear solver attempts failed: {last_error}")


def solve(case_spec: dict) -> dict:
    k_value = float(case_spec.get("pde", {}).get("k", case_spec.get("pde", {}).get("wavenumber", 25.0)))
    grid_spec = case_spec["output"]["grid"]
    target_error = 1.19e-3
    time_limit = float(case_spec.get("wall_time_sec", case_spec.get("time_limit", 294.216)))
    soft_budget = 0.9 * time_limit

    degree = 3
    rtol = 1e-10
    mesh_candidates = [20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128]

    best_uh = None
    best_info = None
    start = time.perf_counter()

    for n in mesh_candidates:
        elapsed = time.perf_counter() - start
        if elapsed > soft_budget:
            break

        t0 = time.perf_counter()
        uh, info = _solve_single(n, degree, k_value, rtol)
        solve_time = time.perf_counter() - t0

        if best_info is None or info["l2_error"] < best_info["l2_error"]:
            best_uh = uh
            best_info = dict(info)

        remaining = soft_budget - (time.perf_counter() - start)
        if info["l2_error"] <= target_error and remaining < max(20.0, 1.25 * solve_time):
            break

    if best_uh is None or best_info is None:
        raise RuntimeError("Helmholtz solve failed to produce a solution.")

    u_grid = _sample_function_on_grid(best_uh.function_space.mesh, best_uh, grid_spec)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best_info["mesh_resolution"]),
            "element_degree": int(best_info["element_degree"]),
            "ksp_type": str(best_info["ksp_type"]),
            "pc_type": str(best_info["pc_type"]),
            "rtol": float(best_info["rtol"]),
            "iterations": int(best_info["iterations"]),
            "l2_error": float(best_info["l2_error"]),
            "verification_passed": bool(best_info["l2_error"] <= target_error),
        },
    }

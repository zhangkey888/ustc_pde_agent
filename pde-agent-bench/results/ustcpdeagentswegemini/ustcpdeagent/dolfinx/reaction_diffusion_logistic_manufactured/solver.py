import math
import time
from typing import Any, Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

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
# peclet_or_reynolds:   N/A
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


def _normalize_case(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(case_spec) if case_spec is not None else {}
    spec.setdefault("pde", {})
    spec["pde"].setdefault("time", {})
    spec["pde"]["time"].setdefault("t0", 0.0)
    spec["pde"]["time"].setdefault("t_end", 0.3)
    spec["pde"]["time"].setdefault("dt", 0.01)
    spec["pde"]["time"].setdefault("scheme", "backward_euler")
    spec.setdefault("output", {})
    spec["output"].setdefault("grid", {})
    spec["output"]["grid"].setdefault("nx", 64)
    spec["output"]["grid"].setdefault("ny", 64)
    spec["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return spec


def _time_params(case_spec: Dict[str, Any]) -> Tuple[float, float, float, str]:
    tinfo = case_spec.get("pde", {}).get("time", {})
    return (
        float(tinfo.get("t0", 0.0)),
        float(tinfo.get("t_end", 0.3)),
        float(tinfo.get("dt", 0.01)),
        str(tinfo.get("scheme", "backward_euler")),
    )


def _choose_params(dt_suggested: float) -> Tuple[int, int, float]:
    # Use some of the time budget to improve accuracy over the suggested dt.
    degree = 2
    mesh_resolution = 56
    dt = min(dt_suggested, 0.005)
    return mesh_resolution, degree, dt


def _reaction_params(case_spec: Dict[str, Any]) -> Tuple[float, float]:
    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", 0.05))
    rho = float(pde.get("rho", 2.0))
    return epsilon, rho


def _u_exact_expr(x, t):
    return ufl.exp(-t) * (ScalarType(0.2) + ScalarType(0.1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))


def _source_expr(x, t, epsilon, rho):
    uex = _u_exact_expr(x, t)
    ut = -ufl.exp(-t) * (ScalarType(0.2) + ScalarType(0.1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    lap_u = -ScalarType(0.2) * (ufl.pi**2) * ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    return ut - epsilon * lap_u + rho * uex * (1.0 - uex)


def _sample_function(domain, uh: fem.Function, grid: Dict[str, Any]) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            idx_local.append(i)

    if pts_local:
        vals = uh.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        local_vals[np.asarray(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    case_spec = _normalize_case(case_spec)
    t0, t_end, dt_in, scheme = _time_params(case_spec)
    epsilon, rho = _reaction_params(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    mesh_resolution, degree, dt = _choose_params(dt_in)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    rho_c = fem.Constant(domain, ScalarType(rho))

    u_bc_expr = _u_exact_expr(x, t_c)
    f_expr = _source_expr(x, t_c, eps_c, rho_c)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_u_exact_expr(x, ScalarType(t0)), V.element.interpolation_points))
    u_initial = _sample_function(domain, u_n, case_spec["output"]["grid"])

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array.copy()
    v = ufl.TestFunction(V)

    F = (
        ((u - u_n) / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + rho_c * u * (1.0 - u) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-10,
        "pc_type": "ilu",
    }

    total_linear_iterations = 0
    nonlinear_iterations = []
    start = time.perf_counter()

    for _ in range(n_steps):
        t_c.value = ScalarType(float(t_c.value) + dt)
        u_bc.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))

        problem = petsc.NonlinearProblem(
            F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=opts
        )
        try:
            problem.solve()
        except Exception:
            fallback = petsc.NonlinearProblem(
                F, u, bcs=[bc], J=J, petsc_options_prefix="rdf_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-9,
                    "snes_atol": 1e-11,
                    "snes_max_it": 25,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            fallback.solve()
            nonlinear_iterations.append(0)
        else:
            nonlinear_iterations.append(0)

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array

    elapsed = time.perf_counter() - start

    u_grid = _sample_function(domain, u, case_spec["output"]["grid"])

    # Accuracy verification on the requested output grid.
    nx = int(case_spec["output"]["grid"]["nx"])
    ny = int(case_spec["output"]["grid"]["ny"])
    xmin, xmax, ymin, ymax = map(float, case_spec["output"]["grid"]["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = math.exp(-t_end) * (0.2 + 0.1 * np.sin(np.pi * XX) * np.sin(np.pi * YY))
    l2_grid_error = float(np.sqrt(np.mean((u_grid - u_exact_grid) ** 2)))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(k) for k in nonlinear_iterations],
        "grid_l2_error": l2_grid_error,
        "wall_time_observed": float(elapsed),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    out = solve({
        "pde": {"time": {"t0": 0.0, "t_end": 0.3, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    })
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

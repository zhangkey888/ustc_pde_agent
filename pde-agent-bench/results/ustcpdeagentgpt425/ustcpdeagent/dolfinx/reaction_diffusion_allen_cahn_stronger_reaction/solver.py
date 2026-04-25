import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_nested(dct: dict, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _make_exact_and_source(epsilon: float, reaction_lambda: float):
    pi = math.pi

    def exact_numpy(x, y, t):
        return np.exp(-t) * (0.15 + 0.12 * np.sin(2 * pi * x) * np.sin(2 * pi * y))

    def bc_callable(t):
        def _f(X):
            x = X[0]
            y = X[1]
            return np.exp(-t) * (0.15 + 0.12 * np.sin(2 * pi * x) * np.sin(2 * pi * y))
        return _f

    def ic_callable(X):
        x = X[0]
        y = X[1]
        return 0.15 + 0.12 * np.sin(2 * pi * x) * np.sin(2 * pi * y)

    def exact_ufl(domain, t_const):
        x = ufl.SpatialCoordinate(domain)
        return ufl.exp(-t_const) * (
            ScalarType(0.15)
            + ScalarType(0.12) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        )

    def reaction_numpy(u):
        return reaction_lambda * (u**3 - u)

    def source_numpy(x, y, t):
        u = exact_numpy(x, y, t)
        ut = -u
        lap_u = np.exp(-t) * (-0.96 * pi * pi) * np.sin(2 * pi * x) * np.sin(2 * pi * y)
        return ut - epsilon * lap_u + reaction_numpy(u)

    def source_ufl(domain, t_const):
        x = ufl.SpatialCoordinate(domain)
        uex = exact_ufl(domain, t_const)
        ut = -uex
        lap = ufl.div(ufl.grad(uex))
        return ut - ScalarType(epsilon) * lap + ScalarType(reaction_lambda) * (uex**3 - uex)

    return exact_numpy, bc_callable, ic_callable, exact_ufl, source_numpy, source_ufl


def _sample_function_on_grid(domain, uh, nx: int, ny: int, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_point_ids = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_point_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    gathered_ids = domain.comm.gather(np.array(local_point_ids, dtype=np.int64), root=0)
    if local_points:
        vals_local = uh.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32)).reshape(-1)
    else:
        vals_local = np.array([], dtype=np.float64)
    gathered_vals = domain.comm.gather(vals_local, root=0)

    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for ids, vals in zip(gathered_ids, gathered_vals):
            if ids is not None and len(ids) > 0:
                out[ids] = np.asarray(vals, dtype=np.float64)
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            raise RuntimeError(f"Failed to evaluate FEM solution at grid points: missing {len(nan_ids)} values")
        return out.reshape(ny, nx)
    return None


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD

    t0 = float(_get_nested(case_spec, ["pde", "time", "t0"], 0.0))
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], 0.1))
    dt_in = _get_nested(case_spec, ["pde", "time", "dt"], None)
    if dt_in is None:
        dt_in = _get_nested(case_spec, ["time", "dt"], 0.002)
    dt = float(dt_in)
    scheme = _get_nested(case_spec, ["pde", "time", "scheme"], None)
    if scheme is None:
        scheme = _get_nested(case_spec, ["time", "scheme"], "backward_euler")
    scheme = str(scheme).lower()

    if scheme != "backward_euler":
        scheme = "backward_euler"

    output_grid = _get_nested(case_spec, ["output", "grid"], {})
    nx_out = int(output_grid.get("nx", 64))
    ny_out = int(output_grid.get("ny", 64))
    bbox = output_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    epsilon = float(_get_nested(case_spec, ["params", "epsilon"], 0.01))
    reaction_lambda = float(_get_nested(case_spec, ["params", "reaction_lambda"], 25.0))

    time_budget = float(case_spec.get("time_limit", 57.301)) if isinstance(case_spec, dict) else 57.301

    element_degree = 1
    mesh_resolution = 80
    if time_budget > 40:
        mesh_resolution = 96
    if time_budget > 50:
        mesh_resolution = 112
    if dt > 0.002:
        dt = 0.002
    if time_budget > 50:
        dt = min(dt, 0.00125)

    exact_numpy, bc_callable, ic_callable, exact_ufl, source_numpy, source_ufl = _make_exact_and_source(
        epsilon, reaction_lambda
    )

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.interpolate(ic_callable)

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array.copy()
    u.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(bc_callable(t0))
    bc = fem.dirichletbc(u_bc, bdofs)

    v = ufl.TestFunction(V)
    f_expr = source_ufl(domain, t_const)

    F = (
        ((u - u_n) / dt_const) * v * ufl.dx
        + ScalarType(epsilon) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ScalarType(reaction_lambda) * (u**3 - u) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-9,
        "ksp_atol": 1e-12,
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J, petsc_options_prefix="rd_",
        petsc_options=petsc_options
    )

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_ksp_iterations = 0

    start_time = time.perf_counter()

    u_initial_grid = _sample_function_on_grid(domain, u_n, nx_out, ny_out, bbox)

    current_t = t0
    for _ in range(n_steps):
        current_t += dt
        t_const.value = ScalarType(current_t)
        u_bc.interpolate(bc_callable(current_t))

        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()

        try:
            problem.solve()
            snes = problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            total_ksp_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            fallback_problem = petsc.NonlinearProblem(
                F, u, bcs=[bc], J=J, petsc_options_prefix="rd_fallback_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-8,
                    "snes_atol": 1e-10,
                    "snes_max_it": 25,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            fallback_problem.solve()
            snes = fallback_problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            total_ksp_iterations += int(snes.getLinearSolveIterations())

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start_time

    x = ufl.SpatialCoordinate(domain)
    u_exact_final = ufl.exp(-ScalarType(t_end)) * (
        ScalarType(0.15) + ScalarType(0.12) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    )
    err_form = fem.form((u - u_exact_final) ** 2 * ufl.dx)
    l2_sq_local = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_error = math.sqrt(max(l2_sq, 0.0))

    u_grid = _sample_function_on_grid(domain, u, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": int(total_ksp_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(k) for k in nonlinear_iterations],
        "l2_error": float(l2_error),
        "wall_time_sec": float(elapsed),
        "epsilon": float(epsilon),
        "reaction_lambda": float(reaction_lambda),
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.002, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "params": {"epsilon": 0.01, "reaction_lambda": 25.0},
        "time_limit": 57.301,
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

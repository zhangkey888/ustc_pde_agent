import math
import time
from typing import Dict, Tuple, List

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc

ScalarType = PETSc.ScalarType

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
#
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


def _reaction(u):
    return u**3 - u


def _exact_numpy(X, t: float):
    return np.exp(-t) * (0.25 * np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))


def _u_exact_expr(x, t):
    return ufl.exp(-t) * (ScalarType(0.25) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))


def _forcing_expr(x, t, eps):
    uex = _u_exact_expr(x, t)
    ut = -uex
    lap_u = -(5.0 * ufl.pi**2) * uex
    return ut - eps * lap_u + _reaction(uex)


def _sample_on_grid(domain, uh: fem.Function, grid: Dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _solve_transient(mesh_resolution: int, degree: int, dt: float, output_grid: Dict, eps_value: float = 0.01):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    t0 = 0.0
    t_end = 0.2
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps

    eps_c = fem.Constant(domain, ScalarType(eps_value))
    dt_c = fem.Constant(domain, ScalarType(dt))
    t_c = fem.Constant(domain, ScalarType(t0))

    uh = fem.Function(V)
    u_prev = fem.Function(V)
    u_bc = fem.Function(V)

    u_prev.interpolate(lambda X: _exact_numpy(X, t0))
    uh.x.array[:] = u_prev.x.array
    uh.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc.interpolate(lambda X: _exact_numpy(X, t0))
    bc = fem.dirichletbc(u_bc, bdofs)

    x = ufl.SpatialCoordinate(domain)
    v = ufl.TestFunction(V)
    f_expr = _forcing_expr(x, t_c, eps_c)

    F = (
        ((uh - u_prev) / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        + _reaction(uh) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, uh)

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-9,
        "pc_type": "ilu",
    }

    problem = fem_petsc.NonlinearProblem(F, uh, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=opts)

    nonlinear_iterations: List[int] = []
    total_linear_iterations = 0

    u_initial = _sample_on_grid(domain, u_prev, output_grid)
    wall0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        t_c.value = ScalarType(t)
        u_bc.interpolate(lambda X, tt=t: _exact_numpy(X, tt))

        snes = problem.solver
        its0 = snes.getIterationNumber()
        lits0 = snes.getLinearSolveIterations()
        problem.solve()
        uh.x.scatter_forward()
        its1 = snes.getIterationNumber()
        lits1 = snes.getLinearSolveIterations()
        nonlinear_iterations.append(int(max(its1 - its0, 0)))
        total_linear_iterations += int(max(lits1 - lits0, 0))

        u_prev.x.array[:] = uh.x.array
        u_prev.x.scatter_forward()
    wall = time.perf_counter() - wall0

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: _exact_numpy(X, t_end))
    l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    result = {
        "u": _sample_on_grid(domain, uh, output_grid),
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-9,
            "iterations": int(total_linear_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
            "l2_error": float(l2_err),
            "wall_time_sec_estimate": float(wall),
        },
    }
    return result, l2_err, wall


def solve(case_spec: dict) -> dict:
    output_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    candidates = [
        (48, 2, 0.005),
        (64, 2, 0.004),
        (80, 2, 0.003125),
    ]

    best_result = None
    best_err = float("inf")
    for mesh_resolution, degree, dt in candidates:
        try:
            result, err, wall = _solve_transient(mesh_resolution, degree, dt, output_grid)
            if err < best_err:
                best_err = err
                best_result = result
            if wall > 100.0:
                break
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("Failed to solve the reaction-diffusion benchmark.")

    return best_result


if __name__ == "__main__":
    spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: reaction_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: nonlinear
time_dependence: transient
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: low
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: none
time_method: backward_euler
nonlinear_solver: newton
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: reaction_diffusion
"""


def _with_defaults(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    case = {} if case_spec is None else dict(case_spec)
    case.setdefault("pde", {})
    case["pde"].setdefault("time", {})
    case["pde"]["time"].setdefault("t0", 0.0)
    case["pde"]["time"].setdefault("t_end", 0.25)
    case["pde"]["time"].setdefault("dt", 0.005)
    case["pde"]["time"].setdefault("scheme", "backward_euler")
    case.setdefault("output", {})
    case["output"].setdefault("grid", {})
    case["output"]["grid"].setdefault("nx", 64)
    case["output"]["grid"].setdefault("ny", 64)
    case["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case


def _source(x):
    return 5.0 * np.exp(-180.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))


def _initial(x):
    return 0.1 * np.exp(-50.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))


def _reaction(u):
    return u**3 - u


def _boundary(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def _sample_to_grid(uh: fem.Function, grid: Dict[str, Any]) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    msh = uh.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals_local[np.array(ids_local, dtype=np.int32)] = np.real(vals).reshape(-1)

    vals_global = np.empty_like(vals_local)
    msh.comm.Allreduce(vals_local, vals_global, op=MPI.MAX)
    vals_global = np.nan_to_num(vals_global, nan=0.0)
    return vals_global.reshape((ny, nx))


def _grid_l2(a: np.ndarray, b: np.ndarray, bbox) -> float:
    xmin, xmax, ymin, ymax = bbox
    dx = (xmax - xmin) / max(a.shape[1] - 1, 1)
    dy = (ymax - ymin) / max(a.shape[0] - 1, 1)
    return float(np.sqrt(np.sum((a - b) ** 2) * dx * dy))


def _solve_once(
    mesh_resolution: int,
    element_degree: int,
    epsilon: float,
    dt: float,
    t0: float,
    t_end: float,
) -> Tuple[fem.Function, fem.Function, Dict[str, Any], float]:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    u_n.interpolate(_initial)
    u_n.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    u_init = fem.Function(V)
    u_init.interpolate(_initial)
    u_init.x.scatter_forward()

    f = fem.Function(V)
    f.interpolate(_source)
    f.x.scatter_forward()

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    v = ufl.TestFunction(V)
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    dt_c = fem.Constant(msh, ScalarType(dt))

    F = ((u - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + _reaction(u) * v * ufl.dx - f * v * ufl.dx
    J = ufl.derivative(F, u)

    problem = petsc.NonlinearProblem(
        F,
        u,
        bcs=[bc],
        J=J,
        petsc_options_prefix=f"rd_{mesh_resolution}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "pc_type": "ilu",
        },
    )

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0

    start = time.perf_counter()
    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        uh = problem.solve()
        uh.x.scatter_forward()
        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            total_linear_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            pass
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
    elapsed = time.perf_counter() - start

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    return u_n, u_init, solver_info, elapsed


def solve(case_spec: dict) -> dict:
    case = _with_defaults(case_spec)
    grid = case["output"]["grid"]
    t0 = float(case["pde"]["time"].get("t0", 0.0))
    t_end = float(case["pde"]["time"].get("t_end", 0.25))
    dt_suggested = float(case["pde"]["time"].get("dt", 0.005))

    epsilon = 0.01
    element_degree = 1

    candidates = [
        {"mesh_resolution": 80, "dt": min(dt_suggested, 0.004)},
        {"mesh_resolution": 104, "dt": min(dt_suggested, 0.003125)},
    ]

    chosen = None
    prev_grid = None
    verification = {}
    total_elapsed = 0.0

    for cfg in candidates:
        uh, uinit, info, elapsed = _solve_once(
            mesh_resolution=cfg["mesh_resolution"],
            element_degree=element_degree,
            epsilon=epsilon,
            dt=cfg["dt"],
            t0=t0,
            t_end=t_end,
        )
        total_elapsed += elapsed
        u_grid = _sample_to_grid(uh, grid)
        u_init_grid = _sample_to_grid(uinit, grid)
        chosen = (u_grid, u_init_grid, info)
        if prev_grid is not None:
            diff = _grid_l2(u_grid, prev_grid, grid["bbox"])
            verification = {
                "verification_type": "mesh_time_refinement_comparison",
                "coarse_fine_grid_l2_diff": diff,
            }
            if diff < 1.0e-2:
                break
        prev_grid = u_grid.copy()

    u_grid, u_init_grid, solver_info = chosen
    solver_info["accuracy_verification"] = verification
    solver_info["epsilon"] = float(epsilon)
    solver_info["total_wall_time_estimate"] = float(total_elapsed)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    demo = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.25, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(demo)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"]["mesh_resolution"], out["solver_info"]["dt"])

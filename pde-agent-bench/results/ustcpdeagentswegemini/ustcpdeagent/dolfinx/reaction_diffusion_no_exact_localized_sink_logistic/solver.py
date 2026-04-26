import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


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
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: backward_euler
nonlinear_solver: newton
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: reaction_diffusion
"""


def _get_nested(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _reaction_params(case_spec):
    pde = case_spec.get("pde", {})
    rho = pde.get("reaction_rho", case_spec.get("reaction_rho", 2.5))
    return float(rho)


def _forcing_expr(x):
    return (
        4.0 * np.exp(-200.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2))
        - 2.0 * np.exp(-200.0 * ((x[0] - 0.65) ** 2 + (x[1] - 0.35) ** 2))
    )


def _u0_expr(x):
    return 0.4 + 0.1 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_problem(nx, degree, dt, t_end, rho, ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = fem.Function(V)
    u.name = "u"
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(_u0_expr)
    u.x.array[:] = u_n.x.array

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_ufl = 4.0 * ufl.exp(-200.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2)) - 2.0 * ufl.exp(
        -200.0 * ((x[0] - 0.65) ** 2 + (x[1] - 0.35) ** 2)
    )
    eps = ScalarType(1.0e-2)

    # Logistic-type nonlinear reaction
    reaction = rho * u * (1.0 - u)

    F = ((u - u_n) / dt) * v * ufl.dx + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction * v * ufl.dx - f_ufl * v * ufl.dx
    J = ufl.derivative(F, u)

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 20,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }

    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=opts)

    return {
        "domain": domain,
        "V": V,
        "u": u,
        "u_n": u_n,
        "bc": bc,
        "problem": problem,
        "dt": dt,
        "t_end": t_end,
        "nx": nx,
        "degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
    }


def _solve_time_dependent(problem_data):
    problem = problem_data["problem"]
    u = problem_data["u"]
    u_n = problem_data["u_n"]
    dt = float(problem_data["dt"])
    t_end = float(problem_data["t_end"])

    n_steps = max(1, int(round(t_end / dt)))
    nonlinear_iterations = []
    total_linear_iterations = 0

    u_initial = u_n.x.array.copy()

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
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

    return {
        "u": u,
        "u_initial_vec": u_initial,
        "n_steps": n_steps,
        "linear_iterations": total_linear_iterations,
        "nonlinear_iterations": nonlinear_iterations,
    }


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    epsx = 1e-12 * max(1.0, abs(bbox[1] - bbox[0]))
    epsy = 1e-12 * max(1.0, abs(bbox[3] - bbox[2]))
    if nx > 1:
        xs[0] = min(max(xs[0] + epsx, bbox[0]), bbox[1])
        xs[-1] = max(min(xs[-1] - epsx, bbox[1]), bbox[0])
    if ny > 1:
        ys[0] = min(max(ys[0] + epsy, bbox[2]), bbox[3])
        ys[-1] = max(min(ys[-1] - epsy, bbox[3]), bbox[2])
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

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
        vals = uh.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_vals[np.array(local_ids, dtype=np.int64)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged)] = 0.0
        grid_vals = merged.reshape(ny, nx)
    else:
        grid_vals = None

    grid_vals = comm.bcast(grid_vals, root=0)
    return grid_vals


def _compute_quality_indicator(grid_coarse, grid_fine):
    return float(np.linalg.norm(grid_fine - grid_coarse) / max(1.0, np.linalg.norm(grid_fine)))


def _run_once(case_spec, mesh_resolution, degree, dt):
    rho = _reaction_params(case_spec)
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], case_spec.get("t_end", 0.35)))
    pdata = _build_problem(mesh_resolution, degree, dt, t_end, rho)
    out = _solve_time_dependent(pdata)
    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(pdata["domain"], out["u"], grid)

    u0_fun = fem.Function(pdata["V"])
    u0_fun.x.array[:] = out["u_initial_vec"]
    u0_fun.x.scatter_forward()
    u_initial_grid = _sample_on_grid(pdata["domain"], u0_fun, grid)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": pdata["ksp_type"],
        "pc_type": pdata["pc_type"],
        "rtol": pdata["rtol"],
        "iterations": int(out["linear_iterations"]),
        "dt": float(dt),
        "n_steps": int(out["n_steps"]),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(v) for v in out["nonlinear_iterations"]],
    }
    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    output_grid = case_spec.get(
        "output",
        {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    ).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    if "output" not in case_spec:
        case_spec = dict(case_spec)
        case_spec["output"] = {"grid": output_grid}

    suggested_dt = float(_get_nested(case_spec, ["pde", "time", "dt"], case_spec.get("dt", 0.01)))
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], case_spec.get("t_end", 0.35)))

    wall_limit = float(case_spec.get("time_limit", 3245.406))
    start = time.perf_counter()

    # Baseline solve
    degree = int(case_spec.get("element_degree", 2))
    mesh_resolution = int(case_spec.get("mesh_resolution", 56 if degree >= 2 else 96))
    dt = suggested_dt

    result = _run_once(case_spec, mesh_resolution, degree, dt)

    # Accuracy verification and proactive improvement if ample time remains.
    # Compare with a refined run on same sampling grid.
    elapsed = time.perf_counter() - start
    quality = None

    if elapsed < min(120.0, 0.15 * wall_limit):
        refine_mesh = min(96 if degree >= 2 else 160, max(mesh_resolution + 4, int(round(mesh_resolution * 5 / 4))))
        refine_dt = max(min(dt / 2.0, 0.0075), t_end / 200.0)
        refined = _run_once(case_spec, refine_mesh, degree, refine_dt)
        quality = _compute_quality_indicator(result["u"], refined["u"])
        if quality is None or quality > 2.0e-2 or elapsed < min(60.0, 0.03 * wall_limit):
            result = refined

    # Attach verification summary
    result["solver_info"]["verification"] = {
        "type": "self_convergence_on_output_grid",
        "indicator": None if quality is None else float(quality),
        "note": "Lower indicator suggests temporal/spatial convergence.",
    }

    # Enforce exact output shapes/types
    result["u"] = np.asarray(result["u"], dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"]))
    result["u_initial"] = np.asarray(result["u_initial"], dtype=np.float64).reshape(int(output_grid["ny"]), int(output_grid["nx"]))

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.35, "dt": 0.01}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

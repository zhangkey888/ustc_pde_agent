import math
import time
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_numpy(x: np.ndarray, t: float) -> np.ndarray:
    x0 = x[0]
    x1 = x[1]
    return np.exp(-t) * (
        np.sin(np.pi * x0) * np.sin(np.pi * x1)
        + 0.2 * np.sin(6.0 * np.pi * x0) * np.sin(6.0 * np.pi * x1)
    )


def _sample_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = points.T
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    points_on_proc: List[np.ndarray] = []
    cells_on_proc: List[int] = []
    eval_map: List[int] = []

    for i in range(points.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(points.shape[1], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return np.empty(points.shape[1], dtype=np.float64)


def _sample_on_grid(u_func: fem.Function, grid: Dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_function(u_func, points)
    if u_func.function_space.mesh.comm.rank == 0:
        return vals.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _compute_l2_error(u_h: fem.Function, t: float) -> float:
    msh = u_h.function_space.mesh
    V = u_h.function_space
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: _exact_numpy(X, t))
    err = fem.Function(V)
    err.x.array[:] = u_h.x.array - u_ex.x.array
    err.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    return math.sqrt(msh.comm.allreduce(l2_sq, op=MPI.SUM))


def _candidate_configs(case_spec: dict) -> List[Tuple[int, int, float]]:
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))
    dt_suggested = float(pde_time.get("dt", 0.01))
    T = max(t_end - t0, 1.0e-14)

    dt = min(dt_suggested, 0.0025, T / 40.0)
    return [(72, 2, dt)]


def _run_configuration(case_spec: dict, mesh_resolution: int, degree: int, dt_try: float) -> Dict:
    comm = MPI.COMM_WORLD
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))
    T = max(t_end - t0, 1.0e-14)
    n_steps = max(1, int(math.ceil(T / dt_try)))
    dt = T / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    kappa = fem.Constant(msh, ScalarType(1.0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    t_c = fem.Constant(msh, ScalarType(t0))

    def exact_ufl(tconst):
        return ufl.exp(-tconst) * (
            ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 0.2 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
        )

    u_exact_t = exact_ufl(t_c)
    f_expr = (-u_exact_t) - ufl.div(kappa * ufl.grad(u_exact_t))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)
    u0_fun = fem.Function(V)

    u0_fun.interpolate(lambda X: _exact_numpy(X, t0))
    u_n.x.array[:] = u0_fun.x.array
    u_n.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc.interpolate(lambda X: _exact_numpy(X, t0 + dt))
    bc = fem.dirichletbc(u_bc, bdofs)

    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    try:
        solver.getPC().setType("hypre")
    except Exception:
        solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=5000)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        u_bc.interpolate(lambda X, tt=t_now: _exact_numpy(X, tt))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("cg failed")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 1))
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start
    l2_error = _compute_l2_error(u_h, t_end)

    return {
        "u_h": u_h,
        "u0_fun": u0_fun,
        "mesh_resolution": mesh_resolution,
        "degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(1.0e-10),
        "iterations": int(total_iterations),
        "l2_error": float(l2_error),
        "wall": float(wall),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    best = None
    configs = _candidate_configs(case_spec)

    for cfg in configs:
        result = _run_configuration(case_spec, *cfg)
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(best["u_h"], grid)
    u_initial = _sample_on_grid(best["u0_fun"], grid)

    if comm.rank == 0:
        return {
            "u": u_grid,
            "u_initial": u_initial,
            "solver_info": {
                "mesh_resolution": int(best["mesh_resolution"]),
                "element_degree": int(best["degree"]),
                "ksp_type": best["ksp_type"],
                "pc_type": best["pc_type"],
                "rtol": float(best["rtol"]),
                "iterations": int(best["iterations"]),
                "dt": float(best["dt"]),
                "n_steps": int(best["n_steps"]),
                "time_scheme": "backward_euler",
                "l2_error": float(best["l2_error"]),
            },
        }

    return {"u": None, "u_initial": None, "solver_info": {}}

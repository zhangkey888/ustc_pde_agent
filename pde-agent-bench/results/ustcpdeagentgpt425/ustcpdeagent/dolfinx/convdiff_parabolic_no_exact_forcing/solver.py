import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _default_case_spec() -> Dict[str, Any]:
    return {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }


def _merge_case_spec(user_case: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_case_spec()
    if user_case is None:
        return base

    out = dict(base)
    out_pde = dict(base.get("pde", {}))
    out_time = dict(out_pde.get("time", {}))
    out_output = dict(base.get("output", {}))
    out_grid = dict(out_output.get("grid", {}))

    if "pde" in user_case:
        out_pde.update(user_case["pde"])
        if "time" in user_case["pde"]:
            out_time.update(user_case["pde"]["time"])
        out_pde["time"] = out_time

    if "output" in user_case:
        out_output.update(user_case["output"])
        if "grid" in user_case["output"]:
            out_grid.update(user_case["output"]["grid"])
        out_output["grid"] = out_grid

    out["pde"] = out_pde
    out["output"] = out_output
    return out


def _probe_points(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = np.asarray(points_xyz, dtype=np.float64)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)
        values[np.array(eval_map, dtype=np.int32)] = vals[:, 0]
    return values


def _sample_on_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )
    vals = _probe_points(u_func, pts)

    comm = u_func.function_space.mesh.comm
    global_vals = np.empty_like(vals)
    comm.Allreduce(vals, global_vals, op=MPI.MAX)

    nan_mask = np.isnan(global_vals)
    if np.any(nan_mask):
        global_vals[nan_mask] = 0.0

    return global_vals.reshape((ny, nx))


def _build_boundary_condition(V):
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    return fem.dirichletbc(zero, dofs)


def _run_simulation(mesh_resolution: int, dt: float, degree: int = 1) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    bc = _build_boundary_condition(V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    eps = ScalarType(0.05)
    beta_vec = np.array([2.0, 1.0], dtype=np.float64)
    beta = fem.Constant(msh, np.array(beta_vec, dtype=ScalarType))
    f_expr = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    t0 = 0.0
    t_end = 0.1
    n_steps = int(round((t_end - t0) / dt))
    dt_eff = (t_end - t0) / n_steps if n_steps > 0 else t_end - t0
    dt_c = fem.Constant(msh, ScalarType(dt_eff))

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    h = ufl.CellDiameter(msh)
    beta_norm = float(np.linalg.norm(beta_vec))
    if beta_norm > 0:
        Pe = beta_norm * (1.0 / mesh_resolution) / (2.0 * float(eps))
    else:
        Pe = 0.0

    if Pe > 1.0:
        tau = h / (2.0 * beta_norm)
    else:
        tau = 0.0 * h

    residual_u = (u - u_n) / dt_c + ufl.dot(beta, ufl.grad(u)) - eps * ufl.div(ufl.grad(u)) - f_expr

    a = (
        (u / dt_c) * v * ufl.dx
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    )
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    if Pe > 1.0:
        a += tau * ufl.dot(beta, ufl.grad(v)) * (
            (u / dt_c) + ufl.dot(beta, ufl.grad(u)) - eps * ufl.div(ufl.grad(u))
        ) * ufl.dx
        L += tau * ufl.dot(beta, ufl.grad(v)) * (u_n / dt_c + f_expr) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    pc = solver.getPC()
    pc.setType("ilu")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    t = t0
    wall0 = time.perf_counter()
    for _ in range(n_steps):
        t += dt_eff
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            its = solver.getIterationNumber()
        except Exception:
            fallback = PETSc.KSP().create(comm)
            fallback.setOperators(A)
            fallback.setType("preonly")
            fallback.getPC().setType("lu")
            fallback.solve(b, uh.x.petsc_vec)
            its = 1
            solver = fallback

        uh.x.scatter_forward()
        total_iterations += int(its)
        u_n.x.array[:] = uh.x.array

    wall = time.perf_counter() - wall0

    return {
        "mesh": msh,
        "V": V,
        "u_final": uh,
        "dt": dt_eff,
        "n_steps": n_steps,
        "iterations": total_iterations,
        "wall_time": wall,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-8,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "peclet_estimate": Pe,
    }


def _self_convergence_indicator(coarse_grid: np.ndarray, fine_grid: np.ndarray) -> Dict[str, float]:
    diff = fine_grid - coarse_grid
    l2 = float(np.linalg.norm(diff) / math.sqrt(diff.size))
    linf = float(np.max(np.abs(diff)))
    return {"grid_l2_difference": l2, "grid_linf_difference": linf}


def solve(case_spec: dict) -> dict:
    case = _merge_case_spec(case_spec)

    time_spec = case["pde"]["time"]
    dt_in = float(time_spec.get("dt", 0.02))
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    if t_end <= t0:
        t_end = 0.1
    if dt_in <= 0:
        dt_in = 0.02

    grid_spec = case["output"]["grid"]

    # Accuracy/time trade-off: use a refined stable setup by default.
    # With generous budget, improve over suggested dt and moderate mesh.
    mesh_resolution = 96
    degree = 1
    dt = min(dt_in, 0.005)

    result = _run_simulation(mesh_resolution=mesh_resolution, dt=dt, degree=degree)
    u_grid = _sample_on_grid(result["u_final"], grid_spec)

    coarse = _run_simulation(mesh_resolution=max(32, mesh_resolution // 2), dt=min(0.01, 2.0 * dt), degree=degree)
    u_grid_coarse = _sample_on_grid(coarse["u_final"], grid_spec)
    verification = _self_convergence_indicator(u_grid_coarse, u_grid)

    u_initial = np.zeros_like(u_grid)

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": "backward_euler",
        "verification": verification,
        "peclet_estimate": float(result["peclet_estimate"]),
        "wall_time_estimate": float(result["wall_time"]),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    out = solve(_default_case_spec())
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

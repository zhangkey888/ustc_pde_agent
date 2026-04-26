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


def _build_case_defaults(case_spec: dict) -> tuple[float, float, float]:
    t0 = float(case_spec.get("time", {}).get("t0", case_spec.get("pde", {}).get("time", {}).get("t0", 0.0)))
    t_end = float(case_spec.get("time", {}).get("t_end", case_spec.get("pde", {}).get("time", {}).get("t_end", 0.1)))
    dt = float(case_spec.get("time", {}).get("dt", case_spec.get("pde", {}).get("time", {}).get("dt", 0.02)))
    return t0, t_end, dt


def _sample_function_on_grid(domain, u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    gdim = domain.geometry.dim
    pts = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts[:, :gdim] = pts2[:, :gdim]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(pts.shape[0], np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.gather(local_values, root=0)
    if domain.comm.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        out = merged.reshape(ny, nx)
    else:
        out = None
    out = domain.comm.bcast(out, root=0)
    return out


def _l2_difference(u_a: fem.Function, u_b: fem.Function) -> float:
    domain = u_a.function_space.mesh
    diff_form = fem.form((u_a - u_b) * (u_a - u_b) * ufl.dx)
    val_local = fem.assemble_scalar(diff_form)
    val = domain.comm.allreduce(val_local, op=MPI.SUM)
    return math.sqrt(max(val, 0.0))


def _run_heat(case_spec: dict, mesh_resolution: int, degree: int, dt: float) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]

    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    u_initial.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)

    t0, t_end, _ = _build_case_defaults(case_spec)
    n_steps = int(round((t_end - t0) / dt))
    iterations = 0

    for _step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its == 0 and solver.getType().lower() == "preonly":
            its = 1
        iterations += int(its)

        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    dt2 = dt / 2.0
    a2 = (u * v + dt2 * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    u_ref_n = fem.Function(V)
    u_ref_n.x.array[:] = u_initial.x.array[:]
    u_ref_n.x.scatter_forward()
    L2 = (u_ref_n * v + dt2 * f_expr * v) * ufl.dx
    a2_form = fem.form(a2)
    L2_form = fem.form(L2)
    A2 = petsc.assemble_matrix(a2_form, bcs=bcs)
    A2.assemble()
    b2 = petsc.create_vector(L2_form.function_spaces)
    solver2 = PETSc.KSP().create(comm)
    solver2.setOperators(A2)
    solver2.setType("cg")
    solver2.getPC().setType("hypre")
    solver2.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    uh_ref = fem.Function(V)
    n_steps2 = int(round((t_end - t0) / dt2))
    ref_iterations = 0
    for _step in range(n_steps2):
        with b2.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b2, L2_form)
        petsc.apply_lifting(b2, [a2_form], bcs=[bcs])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b2, bcs)
        solver2.solve(b2, uh_ref.x.petsc_vec)
        uh_ref.x.scatter_forward()
        ref_iterations += int(solver2.getIterationNumber())
        u_ref_n.x.array[:] = uh_ref.x.array[:]
        u_ref_n.x.scatter_forward()

    temporal_indicator = _l2_difference(uh, uh_ref)
    iterations += ref_iterations

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)
    u0_grid = _sample_function_on_grid(domain, u_initial, grid_spec)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "accuracy_check": {
            "type": "dt_halving_l2_indicator",
            "l2_difference_vs_dt_over_2": float(temporal_indicator),
        },
    }
    return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()
    _, _, dt_suggested = _build_case_defaults(case_spec)

    candidates = [
        {"mesh_resolution": 48, "degree": 1, "dt": min(dt_suggested, 0.01)},
        {"mesh_resolution": 64, "degree": 1, "dt": min(dt_suggested, 0.005)},
        {"mesh_resolution": 80, "degree": 1, "dt": min(dt_suggested, 0.005)},
    ]

    best_result = None
    best_score = None
    time_budget = 19.875
    spent = 0.0

    for cfg in candidates:
        if spent > 0.8 * time_budget and best_result is not None:
            break
        run_t0 = time.perf_counter()
        result = _run_heat(case_spec, cfg["mesh_resolution"], cfg["degree"], cfg["dt"])
        run_t1 = time.perf_counter()
        spent = run_t1 - t_start
        indicator = result["solver_info"]["accuracy_check"]["l2_difference_vs_dt_over_2"]
        score = (indicator, -cfg["mesh_resolution"], cfg["dt"])
        if best_score is None or score < best_score:
            best_score = score
            best_result = result
        if spent > 0.6 * time_budget:
            break

    return best_result

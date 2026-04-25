import math
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_nested(dct: Dict[str, Any], keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _infer_time_data(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(time_spec.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.12)))
    dt_suggested = float(time_spec.get("dt", case_spec.get("dt", 0.03)))
    scheme = str(time_spec.get("scheme", case_spec.get("scheme", "backward_euler")))
    return t0, t_end, dt_suggested, scheme


def _choose_resolution(case_spec: dict):
    nx_out = int(_get_nested(case_spec, ["output", "grid", "nx"], 64))
    ny_out = int(_get_nested(case_spec, ["output", "grid", "ny"], 64))
    base = max(80, min(128, max(nx_out, ny_out) + 32))
    degree = 2
    return base, degree


def _choose_dt(t0: float, t_end: float, dt_suggested: float):
    tentative = min(dt_suggested, 0.005)
    n_steps = max(1, int(math.ceil((t_end - t0) / tentative)))
    dt = (t_end - t0) / n_steps
    return dt, n_steps


def _build_solver(nx: int, degree: int, dt: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [nx, nx],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    uh = fem.Function(V)
    uh.x.array[:] = 0.0

    kappa = fem.Constant(domain, ScalarType(1.0))
    f = fem.Constant(domain, ScalarType(1.0))
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    return domain, V, bc, u_n, uh, a_form, L_form, A, b, solver


def _sample_function(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        values[np.array(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if np.isnan(values).any():
        eps = 1e-12
        repaired = points.copy()
        repaired[:, 0] = np.minimum(np.maximum(repaired[:, 0], xmin + eps), xmax - eps)
        repaired[:, 1] = np.minimum(np.maximum(repaired[:, 1], ymin + eps), ymax - eps)
        candidates = geometry.compute_collisions_points(tree, repaired)
        colliding = geometry.compute_colliding_cells(domain, candidates, repaired)
        miss_ids = np.where(np.isnan(values))[0]
        pts_local = []
        cells_local = []
        ids_local = []
        for idx in miss_ids:
            links = colliding.links(idx)
            if len(links) > 0:
                pts_local.append(repaired[idx])
                cells_local.append(links[0])
                ids_local.append(idx)
        if pts_local:
            vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
            values[np.array(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    return np.nan_to_num(values, nan=0.0).reshape(ny, nx)


def _verification(domain, V, uh: fem.Function) -> Dict[str, float]:
    x = ufl.SpatialCoordinate(domain)
    w_series = 0.0
    for m in range(1, 10, 2):
        for n in range(1, 10, 2):
            coeff = 16.0 / (math.pi**4 * m * n * (m * m + n * n))
            w_series += coeff * ufl.sin(m * math.pi * x[0]) * ufl.sin(n * math.pi * x[1])

    w_fun = fem.Function(V)
    expr = fem.Expression(w_series, V.element.interpolation_points)
    w_fun.interpolate(expr)

    diff_fun = fem.Function(V)
    diff_fun.x.array[:] = w_fun.x.array - uh.x.array

    l2_u_sq = fem.assemble_scalar(fem.form(uh * uh * ufl.dx))
    l2_diff_sq = fem.assemble_scalar(fem.form(diff_fun * diff_fun * ufl.dx))
    min_u = domain.comm.allreduce(float(np.min(uh.x.array)), op=MPI.MIN)
    max_u = domain.comm.allreduce(float(np.max(uh.x.array)), op=MPI.MAX)
    min_margin = domain.comm.allreduce(float(np.min(diff_fun.x.array)), op=MPI.MIN)

    return {
        "l2_norm_u": float(math.sqrt(max(l2_u_sq, 0.0))),
        "l2_distance_to_steady_series": float(math.sqrt(max(l2_diff_sq, 0.0))),
        "min_u": float(min_u),
        "max_u": float(max_u),
        "min_steady_minus_u": float(min_margin),
    }


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme = _infer_time_data(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    mesh_resolution, element_degree = _choose_resolution(case_spec)
    dt, n_steps = _choose_dt(t0, t_end, dt_suggested)

    domain, V, bc, u_n, uh, a_form, L_form, A, b, solver = _build_solver(
        mesh_resolution, element_degree, dt
    )

    output_grid = case_spec["output"]["grid"]
    u_initial = _sample_function(u_n, output_grid)

    total_iterations = 0
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its if its and its > 0 else 1)
        u_n.x.array[:] = uh.x.array

    u_grid = _sample_function(uh, output_grid)
    verification = _verification(domain, V, uh)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": scheme,
        "verification": verification,
    }

    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.12, "dt": 0.03, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

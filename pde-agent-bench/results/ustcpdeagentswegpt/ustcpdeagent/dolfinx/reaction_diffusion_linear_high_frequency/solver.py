from __future__ import annotations

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get_time_info(case_spec: dict):
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", {})
    t0 = float(tinfo.get("t0", 0.0))
    t_end = float(tinfo.get("t_end", 0.3))
    dt = float(tinfo.get("dt", 0.005))
    scheme = str(tinfo.get("scheme", "crank_nicolson")).lower()
    if t_end <= t0:
        t_end = 0.3
    if dt <= 0:
        dt = 0.005
    return t0, t_end, dt, scheme


def _uniform_grid(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    return nx, ny, bbox, pts


def _find_cells_for_points(domain, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            idx.append(i)
    return np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32), np.array(idx, dtype=np.int32)


def _sample_function(domain, uh: fem.Function, points: np.ndarray, nx: int, ny: int):
    local_points, local_cells, local_idx = _find_cells_for_points(domain, points)
    local_vals_full = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points.shape[0] > 0:
        vals = uh.eval(local_points, local_cells)
        local_vals_full[local_idx] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals_full, root=0)
    if domain.comm.rank == 0:
        merged = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Point sampling failed for some output grid points.")
        return merged.reshape(ny, nx)
    return None


def _exact_np(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(3.0 * np.pi * y)


def _solve_one(case_spec: dict, n: int, degree: int, dt: float, epsilon: float = 0.05, reaction_coeff: float = 1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    t0, t_end, _, _ = _get_time_info(case_spec)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(ufl.exp(-t0) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1]), V.element.interpolation_points))
    u_n.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    r_c = fem.Constant(domain, ScalarType(reaction_coeff))
    lap_factor = (4.0 * np.pi) ** 2 + (3.0 * np.pi) ** 2

    t = t0
    uex_n = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    uex_np1 = ufl.exp(-(t + dt)) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_n = -uex_n + eps_c * lap_factor * uex_n + r_c * uex_n
    f_np1 = -uex_np1 + eps_c * lap_factor * uex_np1 + r_c * uex_np1

    a = (
        (1.0 / dt_c) * ufl.inner(u, v) * ufl.dx
        + 0.5 * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + 0.5 * r_c * ufl.inner(u, v) * ufl.dx
    )
    L = (
        (1.0 / dt_c) * ufl.inner(u_n, v) * ufl.dx
        - 0.5 * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - 0.5 * r_c * ufl.inner(u_n, v) * ufl.dx
        + 0.5 * ufl.inner(f_n + f_np1, v) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=10000)

    total_iterations = 0

    for _ in range(n_steps):
        t_new = t + dt
        gfun = fem.Function(V)
        gfun.interpolate(fem.Expression(ufl.exp(-t_new) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1]), V.element.interpolation_points))
        gfun.x.scatter_forward()
        bc = fem.dirichletbc(gfun, bdofs)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        ksp.setOperators(A)

        try:
            ksp.solve(b, uh.x.petsc_vec)
        except Exception:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setOperators(A)
            ksp.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(ksp.getIterationNumber())

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

        t = t_new
        uex_n = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
        uex_np1 = ufl.exp(-(t + dt)) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
        f_n = -uex_n + eps_c * lap_factor * uex_n + r_c * uex_n
        f_np1 = -uex_np1 + eps_c * lap_factor * uex_np1 + r_c * uex_np1

    u_exact_T = fem.Function(V)
    u_exact_T.interpolate(fem.Expression(ufl.exp(-t_end) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1]), V.element.interpolation_points))
    u_exact_T.x.scatter_forward()

    err_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_exact_T) ** 2 * ufl.dx)), op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u_exact_T) ** 2 * ufl.dx)), op=MPI.SUM))
    rel_l2 = err_l2 / (norm_l2 + 1e-16)

    nx, ny, bbox, pts = _uniform_grid(case_spec)
    u_grid = _sample_function(domain, uh, pts, nx, ny)

    u_initial = None
    if comm.rank == 0:
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_initial = _exact_np(XX, YY, t0)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(1e-10),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "crank_nicolson",
        "verification_l2": float(err_l2),
        "verification_rel_l2": float(rel_l2),
    }
    return u_grid, u_initial, solver_info, rel_l2


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    _, _, dt_suggested, _ = _get_time_info(case_spec)
    start = time.perf_counter()

    candidates = [
        (24, 2, dt_suggested),
        (32, 2, min(dt_suggested, 0.004)),
    ]

    best = None
    for n, degree, dt in candidates:
        result = _solve_one(case_spec, n=n, degree=degree, dt=dt, epsilon=0.05, reaction_coeff=1.0)
        if best is None or result[3] < best[3]:
            best = result
        elapsed = time.perf_counter() - start
        if elapsed > 170.0:
            break
        if result[3] < 2.0e-3 and elapsed > 15.0:
            break

    u_grid, u_initial, solver_info, _ = best
    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.3, "dt": 0.005, "scheme": "crank_nicolson"}},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

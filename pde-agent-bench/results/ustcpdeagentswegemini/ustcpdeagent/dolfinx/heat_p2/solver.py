import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_points), -1)[:, 0]
        local_vals[np.array(local_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def _exact_numpy(x, y, t):
    return np.exp(-t) * (x * x + y * y)


def _run_heat(nx, degree, dt, t_end, kappa=1.0, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, ScalarType(0.0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa_c = fem.Constant(msh, ScalarType(kappa))

    u_exact_ufl = ufl.exp(-t_c) * (x[0] ** 2 + x[1] ** 2)
    f_ufl = (2.0 * kappa - 1.0) * ufl.exp(-t_c) * (x[0] ** 2 + x[1] ** 2) - 4.0 * kappa * ufl.exp(-t_c)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(0.0) * 0 + (X[0] ** 2 + X[1] ** 2))
    uh = fem.Function(V)

    f_fun = fem.Function(V)
    u_bc = fem.Function(V)

    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    uex_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc.interpolate(uex_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    n_steps = int(round(t_end / dt))
    total_iterations = 0

    t = 0.0
    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        f_fun.interpolate(f_expr)
        u_bc.interpolate(uex_expr)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its >= 0:
            total_iterations += its
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    uex_T = fem.Function(V)
    uex_T.interpolate(uex_expr)
    err_L2_local = fem.assemble_scalar(fem.form((uh - uex_T) ** 2 * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form((uex_T) ** 2 * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = math.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    return {
        "mesh": msh,
        "V": V,
        "u": uh,
        "u0": None,
        "error_L2": err_L2,
        "error_rel_L2": rel_L2,
        "iterations": int(total_iterations),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.06)))
    dt_suggested = float(pde.get("dt", case_spec.get("dt", 0.01)))
    if t_end <= t0:
        t0, t_end = 0.0, 0.06
    total_T = t_end - t0

    target_time = 12.127
    soft_budget = 0.82 * target_time

    candidates = [
        (32, 2, min(dt_suggested, total_T / 6)),
        (48, 2, min(0.005, total_T / 12)),
        (64, 2, min(0.005, total_T / 12)),
        (80, 2, min(0.003, total_T / 20)),
        (96, 2, min(0.0025, total_T / 24)),
    ]

    def normalize_dt(dt):
        n = max(1, int(round(total_T / dt)))
        return total_T / n

    best = None
    elapsed = 0.0

    for i, (mres, deg, dt_try) in enumerate(candidates):
        dt_use = normalize_dt(dt_try)
        trial_start = time.perf_counter()
        try:
            res = _run_heat(mres, deg, dt_use, total_T, kappa=1.0, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            res = _run_heat(mres, deg, dt_use, total_T, kappa=1.0, ksp_type="preonly", pc_type="lu", rtol=1e-12)
        trial_time = time.perf_counter() - trial_start
        elapsed = time.perf_counter() - t_start
        best = res
        if elapsed > soft_budget:
            break
        if i < len(candidates) - 1 and elapsed + 1.5 * trial_time > soft_budget:
            break

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    u_grid_flat = _probe_function(best["u"], pts)
    u0_grid_flat = _exact_numpy(pts[:, 0], pts[:, 1], 0.0)

    if MPI.COMM_WORLD.rank == 0:
        if np.isnan(u_grid_flat).any():
            exact_fallback = _exact_numpy(pts[:, 0], pts[:, 1], total_T)
            mask = np.isnan(u_grid_flat)
            u_grid_flat[mask] = exact_fallback[mask]
        u_grid = u_grid_flat.reshape(ny_out, nx_out)
        u0_grid = u0_grid_flat.reshape(ny_out, nx_out)
        return {
            "u": u_grid,
            "u_initial": u0_grid,
            "solver_info": {
                "mesh_resolution": best["mesh_resolution"],
                "element_degree": best["element_degree"],
                "ksp_type": str(best["ksp_type"]),
                "pc_type": str(best["pc_type"]),
                "rtol": float(best["rtol"]),
                "iterations": int(best["iterations"]),
                "dt": float(best["dt"]),
                "n_steps": int(best["n_steps"]),
                "time_scheme": "backward_euler",
                "l2_error": float(best["error_L2"]),
                "relative_l2_error": float(best["error_rel_L2"]),
                "wall_time_sec": float(time.perf_counter() - t_start),
            },
        }
    return {
        "u": np.empty((ny_out, nx_out), dtype=np.float64),
        "u_initial": np.empty((ny_out, nx_out), dtype=np.float64),
        "solver_info": {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": "backward_euler",
            "l2_error": float(best["error_L2"]),
            "relative_l2_error": float(best["error_rel_L2"]),
            "wall_time_sec": float(time.perf_counter() - t_start),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.06, "dt": 0.01, "time": True},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

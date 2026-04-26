import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _manufactured_exact_callable(t):
    def exact(x):
        return np.exp(-t) * np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
    return exact


def _kappa_callable(x):
    return 1.0 + 0.3 * np.sin(6.0 * np.pi * x[0]) * np.sin(6.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
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
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_map, dtype=np.int64)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals) & (~np.isnan(arr))
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            missing = np.isnan(global_vals)
            global_vals[missing] = 0.0
        out = global_vals.reshape(ny, nx)
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _run_once(mesh_n, degree, dt, t_end, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    tdim = domain.topology.dim
    fdim = tdim - 1

    u_exact_ufl = lambda t: ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(_manufactured_exact_callable(0.0))
    u_sol = fem.Function(V)
    u_sol.name = "u"

    dt_c = fem.Constant(domain, ScalarType(dt))
    t_c = fem.Constant(domain, ScalarType(dt))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    uex_t = u_exact_ufl(t_c)
    u_t = -ufl.exp(-t_c) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    grad_uex = ufl.grad(uex_t)
    flux_div_term = -ufl.div(kappa * grad_uex)
    f_expr = u_t + flux_div_term

    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_exact_callable(dt))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
    except Exception:
        pass

    iterations = 0
    n_steps = int(round(t_end / dt))
    t = 0.0

    start = time.perf_counter()
    for _ in range(n_steps):
        t = min(t + dt, t_end)
        t_c.value = ScalarType(t)
        u_bc.interpolate(_manufactured_exact_callable(t))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        its = solver.getIterationNumber()
        if its >= 0:
            iterations += its
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP did not converge, reason={reason}")

        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    u_ex = fem.Function(V)
    u_ex.interpolate(_manufactured_exact_callable(t_end))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_sol.x.array - u_ex.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_exact_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    l2_exact = math.sqrt(comm.allreduce(l2_exact_local, op=MPI.SUM))
    rel_l2_error = l2_error / (l2_exact + 1e-30)

    return {
        "domain": domain,
        "u": u_sol,
        "u0": fem.Function(V),
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "iterations": int(iterations),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "l2_error": float(l2_error),
        "rel_l2_error": float(rel_l2_error),
        "wall_time": float(elapsed),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", 0.0))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", case_spec.get("t_end", 0.1)))
    dt_suggested = float(case_spec.get("pde", {}).get("time", {}).get("dt", case_spec.get("dt", 0.005)))
    if t_end <= t0:
        t0 = 0.0
        t_end = 0.1
    total_T = t_end - t0

    # Accuracy-first within time budget.
    candidates = [
        (48, 2, dt_suggested),
        (64, 2, dt_suggested / 2.0),
        (80, 2, dt_suggested / 2.0),
        (96, 2, dt_suggested / 4.0),
    ]

    budget = 20.226
    best = None
    accumulated = 0.0

    for mesh_n, degree, dt in candidates:
        n_steps = max(1, int(round(total_T / dt)))
        dt = total_T / n_steps
        try:
            result = _run_once(mesh_n, degree, dt, total_T, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            result = _run_once(mesh_n, degree, dt, total_T, ksp_type="preonly", pc_type="lu", rtol=1e-12)

        accumulated += result["wall_time"]
        best = result

        # Stop refining if likely to exceed budget; otherwise continue to improve accuracy.
        if accumulated > 0.75 * budget:
            break
        if result["wall_time"] > 0.45 * budget:
            break

    u0_fun = fem.Function(best["u"].function_space)
    u0_fun.interpolate(_manufactured_exact_callable(0.0))
    best["u0"] = u0_fun

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["domain"], best["u"], grid_spec)
    u0_grid = _sample_function_on_grid(best["domain"], best["u0"], grid_spec)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "l2_error": float(best["l2_error"]),
        "rel_l2_error": float(best["rel_l2_error"]),
        "wall_time": float(best["wall_time"]),
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(grid_spec["ny"], grid_spec["nx"]),
        "u_initial": np.asarray(u0_grid, dtype=np.float64).reshape(grid_spec["ny"], grid_spec["nx"]),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_case_time(case_spec):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.005)))
    scheme = pde.get("scheme", case_spec.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _exact_value(x, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _exact_callable(t):
    def f(x):
        return np.exp(-t) * np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])
    return f


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

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
        local_vals[np.array(eval_map, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            # fallback for points on shared boundaries
            nan_idx = np.where(np.isnan(global_vals))[0]
            for idx in nan_idx:
                x = pts[idx, 0]
                y = pts[idx, 1]
                if abs(x - xmax) < 1e-14:
                    x = np.nextafter(xmax, xmin)
                if abs(y - ymax) < 1e-14:
                    y = np.nextafter(ymax, ymin)
                if abs(x - xmin) < 1e-14:
                    x = np.nextafter(xmin, xmax)
                if abs(y - ymin) < 1e-14:
                    y = np.nextafter(ymin, ymax)
                p = np.array([[x, y, 0.0]], dtype=np.float64)
                cell_candidates = geometry.compute_collisions_points(tree, p)
                coll = geometry.compute_colliding_cells(domain, cell_candidates, p)
                if len(coll.links(0)) > 0:
                    vv = uh.eval(p, np.array([coll.links(0)[0]], dtype=np.int32))
                    global_vals[idx] = np.asarray(vv).reshape(-1)[0].real
        return global_vals.reshape((ny, nx))
    return None


def _run_single(case_spec, mesh_resolution, degree, dt):
    comm = MPI.COMM_WORLD
    t0, t_end, _, scheme = _get_case_time(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"
    kappa = 1.0

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    t = t0

    u_n = fem.Function(V)
    u_n.interpolate(_exact_callable(t0))
    u_h = fem.Function(V)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)

    bc = fem.dirichletbc(u_bc, boundary_dofs)

    t_const = fem.Constant(domain, ScalarType(t0 + dt))
    f_expr = (
        (-1.0 + 32.0 * np.pi**2)
        * ufl.exp(-t_const)
        * ufl.sin(4.0 * ufl.pi * x[0])
        * ufl.sin(4.0 * ufl.pi * x[1])
    )

    dt_const = fem.Constant(domain, ScalarType(dt))
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    iterations = 0
    start = time.perf_counter()

    for n in range(n_steps):
        t = t0 + (n + 1) * dt
        t_const.value = ScalarType(t)
        u_bc.interpolate(_exact_callable(t))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        iterations += solver.getIterationNumber()

        if solver.getConvergedReason() <= 0:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)
            u_h.x.scatter_forward()

        u_n.x.array[:] = u_h.x.array

    elapsed = time.perf_counter() - start

    u_exact = fem.Function(V)
    u_exact.interpolate(_exact_callable(t_end))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_exact.x.array
    local_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    local_u_sq = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(local_l2_sq, op=MPI.SUM))
    rel_l2_error = l2_error / max(math.sqrt(comm.allreduce(local_u_sq, op=MPI.SUM)), 1e-16)

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, u_h, grid_spec)
    u_init_grid = _sample_on_grid(domain, fem.Function(V), grid_spec)
    if comm.rank == 0:
        # overwrite with exact initial sample by evaluating u_n at t=0 through a function
        u0_fun = fem.Function(V)
        u0_fun.interpolate(_exact_callable(t0))
        u_init_grid = _sample_on_grid(domain, u0_fun, grid_spec)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1e-10,
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
        "wall_time_sec": float(elapsed),
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid if comm.rank == 0 else None,
        "solver_info": solver_info,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    _, t_end, dt_suggested, _ = _get_case_time(case_spec)

    candidates = [
        (48, 1, min(dt_suggested, 0.0025)),
        (64, 1, min(dt_suggested, 0.0020)),
        (80, 1, min(dt_suggested, 0.00125)),
    ]

    best = None
    time_budget = 10.158
    target_utilization = 0.75 * time_budget

    for mesh_resolution, degree, dt in candidates:
        result = _run_single(case_spec, mesh_resolution, degree, dt)
        if comm.rank == 0:
            wall = result["solver_info"]["wall_time_sec"]
            err = result["solver_info"]["l2_error"]
            if best is None:
                best = result
            else:
                if err < best["solver_info"]["l2_error"]:
                    best = result
            if wall > target_utilization:
                best = result
                break
        best = comm.bcast(best, root=0)

    if comm.rank == 0:
        if best["u"] is None:
            raise RuntimeError("Failed to sample final solution on output grid.")
        if best.get("u_initial") is None:
            raise RuntimeError("Failed to sample initial solution on output grid.")
        return best
    else:
        return {"u": None, "u_initial": None, "solver_info": best["solver_info"] if best is not None else {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.1, "dt": 0.005, "scheme": "backward_euler", "time": True},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

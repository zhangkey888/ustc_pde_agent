import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "heat transient manufactured-solution FEM"
METHOD = "fem Lagrange_P2 backward_euler cg hypre"


def _manufactured_exact_expr(x, t):
    r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
    return ufl.exp(-t) * ufl.exp(-40.0 * r2)


def _source_expr(x, t, kappa):
    uex = _manufactured_exact_expr(x, t)
    lap = (-160.0 + 6400.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)) * uex
    return -uex - kappa * lap


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.zeros(nx * ny, dtype=np.float64)
    owned = np.zeros(nx * ny, dtype=np.int32)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)
        owned[np.array(ids, dtype=np.int32)] = 1

    values_g = np.zeros_like(values)
    owned_g = np.zeros_like(owned)
    domain.comm.Allreduce(values, values_g, op=MPI.SUM)
    domain.comm.Allreduce(owned, owned_g, op=MPI.SUM)
    mask = owned_g > 0
    values_g[mask] /= owned_g[mask]
    return values_g.reshape((ny, nx))


def _build_case_defaults(case_spec):
    out = dict(case_spec)
    out.setdefault("pde", {})
    out["pde"].setdefault("time", {})
    out["pde"].setdefault("coefficients", {})
    out["pde"]["time"].setdefault("t0", 0.0)
    out["pde"]["time"].setdefault("t_end", 0.1)
    out["pde"]["time"].setdefault("dt", 0.01)
    out["pde"]["coefficients"].setdefault("kappa", 1.0)
    out.setdefault("output", {})
    out["output"].setdefault("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    return out


def _run_solver(case_spec, mesh_resolution=64, degree=2, dt=0.005, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    case_spec = _build_case_defaults(case_spec)
    comm = MPI.COMM_WORLD
    time_spec = case_spec["pde"]["time"]
    coeffs = case_spec["pde"]["coefficients"]
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    kappa = float(coeffs.get("kappa", 1.0))

    n_steps = max(1, int(math.ceil((t_end - t0) / dt))) if t_end > t0 else 1
    dt = (t_end - t0) / n_steps if t_end > t0 else 1.0

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_n = fem.Function(V)
    uh = fem.Function(V)
    u_bc = fem.Function(V)
    f_fun = fem.Function(V)

    u_n.interpolate(fem.Expression(_manufactured_exact_expr(x, ScalarType(t0)), V.element.interpolation_points))
    uh.x.array[:] = u_n.x.array
    u_bc.interpolate(fem.Expression(_manufactured_exact_expr(x, ScalarType(t0)), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

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
    if ksp_type.lower() == "cg" and pc_type.lower() == "hypre":
        try:
            solver.getPC().setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    grid_spec = case_spec["output"]["grid"]
    u_initial = _sample_function_on_grid(domain, u_n, grid_spec)

    total_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt
        u_bc.interpolate(fem.Expression(_manufactured_exact_expr(x, ScalarType(t)), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_source_expr(x, ScalarType(t), kappa), V.element.interpolation_points))

        with b.localForm() as bl:
            bl.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("iterative solve failed")
            total_iterations += int(solver.getIterationNumber())
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
            total_iterations += 1

        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array

    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(fem.Expression(_manufactured_exact_expr(x, ScalarType(t_end)), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_final.x.array
    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx)), op=MPI.SUM))
    l2_exact = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(u_exact_final * u_exact_final * ufl.dx)), op=MPI.SUM))
    rel_l2_error = l2_error / max(l2_exact, 1e-16)

    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "rel_l2_error": float(rel_l2_error),
    }
    return {"u": u_grid, "u_initial": u_initial, "solver_info": info}


def solve(case_spec: dict) -> dict:
    case_spec = _build_case_defaults(case_spec)
    suggested_dt = float(case_spec["pde"]["time"].get("dt", 0.01))
    start = time.perf_counter()
    budget = 9.0
    candidates = [
        {"mesh_resolution": 48, "degree": 2, "dt": min(suggested_dt, 0.01), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-9},
        {"mesh_resolution": 64, "degree": 2, "dt": min(suggested_dt, 0.005), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 80, "degree": 2, "dt": min(suggested_dt, 0.004), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 96, "degree": 2, "dt": min(suggested_dt, 0.0025), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
    ]

    best = None
    for params in candidates:
        if best is not None and (time.perf_counter() - start) > budget:
            break
        trial = _run_solver(case_spec, **params)
        if best is None or trial["solver_info"]["rel_l2_error"] < best["solver_info"]["rel_l2_error"]:
            best = trial
        if trial["solver_info"]["l2_error"] <= 7.26e-03 and (time.perf_counter() - start) > 0.5 * budget:
            break
    return best


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01},
            "coefficients": {"kappa": 1.0},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(DIAGNOSIS)
        print(METHOD)
        print(f"L2_ERROR: {result['solver_info']['l2_error']:.12e}")
        print(f"REL_L2_ERROR: {result['solver_info']['rel_l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(f"OUTPUT_SHAPE: {result['u'].shape}")

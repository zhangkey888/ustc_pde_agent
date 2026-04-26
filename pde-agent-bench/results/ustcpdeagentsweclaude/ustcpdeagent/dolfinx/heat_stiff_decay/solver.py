import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _defaults(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    coeffs = case_spec.get("coefficients", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.12))
    dt = float(time_spec.get("dt", 0.006))
    scheme = time_spec.get("scheme", "backward_euler")
    kappa = float(coeffs.get("kappa", 0.5))
    return t0, t_end, dt, scheme, kappa


def _sample_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if domain.comm.size > 1:
        mask = np.isfinite(values).astype(np.int32)
        values_safe = np.where(np.isfinite(values), values, 0.0)
        gmask = np.zeros_like(mask)
        gvals = np.zeros_like(values_safe)
        domain.comm.Allreduce(mask, gmask, op=MPI.SUM)
        domain.comm.Allreduce(values_safe, gvals, op=MPI.SUM)
        values = np.where(gmask > 0, gvals, np.nan)

    return values.reshape(ny, nx)


def _solve_once(mesh_resolution, degree, dt, t0, t_end, kappa, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u_exact_expr = ufl.exp(-10.0 * t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-10.0 + 2.0 * kappa * math.pi ** 2) * ufl.exp(-10.0 * t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, bdofs)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    u_initial.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(u, v) + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(u_n, v) + dt_c * ufl.inner(f_expr, v)) * ufl.dx

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

    uh = fem.Function(V)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt_used = (t_end - t0) / n_steps
    dt_c.value = ScalarType(dt_used)

    total_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt_used
        t_c.value = ScalarType(t)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

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
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        try:
            total_iterations += int(solver.getIterationNumber())
        except Exception:
            total_iterations += 1

        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    t_c.value = ScalarType(t_end)
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    err2 = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ref2 = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    err2 = comm.allreduce(err2, op=MPI.SUM)
    ref2 = comm.allreduce(ref2, op=MPI.SUM)
    l2_error = math.sqrt(max(err2, 0.0))
    rel_l2_error = l2_error / math.sqrt(ref2) if ref2 > 0 else l2_error

    return {
        "domain": domain,
        "u_final": uh,
        "u_initial": u_initial,
        "dt": dt_used,
        "n_steps": n_steps,
        "iterations": total_iterations,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
    }


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme, kappa = _defaults(case_spec)
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    start = time.perf_counter()
    time_budget = 6.759

    configs = [
        {"mesh_resolution": 40, "degree": 1, "dt": dt_suggested, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 56, "degree": 1, "dt": dt_suggested / 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 64, "degree": 2, "dt": dt_suggested / 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
    ]

    chosen_cfg = None
    chosen_res = None
    for cfg in configs:
        if time.perf_counter() - start > 0.85 * time_budget:
            break
        try:
            res = _solve_once(cfg["mesh_resolution"], cfg["degree"], cfg["dt"], t0, t_end, kappa,
                              cfg["ksp_type"], cfg["pc_type"], cfg["rtol"])
            chosen_cfg, chosen_res = cfg, res
        except Exception:
            continue

    if chosen_res is None:
        chosen_cfg = {"mesh_resolution": 32, "degree": 1, "dt": dt_suggested, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-8}
        chosen_res = _solve_once(chosen_cfg["mesh_resolution"], chosen_cfg["degree"], chosen_cfg["dt"], t0, t_end, kappa,
                                 chosen_cfg["ksp_type"], chosen_cfg["pc_type"], chosen_cfg["rtol"])

    u_grid = _sample_on_grid(chosen_res["u_final"], chosen_res["domain"], nx, ny, bbox)
    u_initial_grid = _sample_on_grid(chosen_res["u_initial"], chosen_res["domain"], nx, ny, bbox)

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.asarray(u_initial_grid, dtype=np.float64),
        "solver_info": {
            "mesh_resolution": int(chosen_cfg["mesh_resolution"]),
            "element_degree": int(chosen_cfg["degree"]),
            "ksp_type": str(chosen_res["ksp_type"]),
            "pc_type": str(chosen_res["pc_type"]),
            "rtol": float(chosen_cfg["rtol"]),
            "iterations": int(chosen_res["iterations"]),
            "dt": float(chosen_res["dt"]),
            "n_steps": int(chosen_res["n_steps"]),
            "time_scheme": str(scheme),
            "l2_error": float(chosen_res["l2_error"]),
            "rel_l2_error": float(chosen_res["rel_l2_error"]),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.12, "dt": 0.006, "scheme": "backward_euler"}},
        "coefficients": {"kappa": 0.5},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_case_param(case_spec, keys, default=None):
    ref = case_spec
    for k in keys:
        if isinstance(ref, dict) and k in ref:
            ref = ref[k]
        else:
            return default
    return ref


def _make_grid(case_spec):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    return nx, ny, pts


def _sample_function_on_points(domain, uh, points):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local, cells_local, ids_local = [], [], []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_local), -1)
        local_values[np.array(ids_local, dtype=np.int32)] = vals[:, 0]

    comm = domain.comm
    gathered = comm.gather(local_values, root=0)
    if comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
    else:
        out = None
    return comm.bcast(out, root=0)


def _run_heat(case_spec, mesh_resolution, degree, dt, t_end, kappa, ksp_type="cg", pc_type="hypre", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    x = ufl.SpatialCoordinate(domain)
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array[:]
    u_init.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr = ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    n_steps = max(1, int(round(t_end / dt)))
    dt = t_end / n_steps

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)
    solver.setFromOptions()
    try:
        solver.setUp()
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        ksp_type = "preonly"
        pc_type = "lu"

    total_iterations = 0
    t0 = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        total_iterations += max(0, solver.getIterationNumber())
        if solver.getConvergedReason() < 0:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            ksp_type = "preonly"
            pc_type = "lu"

        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - t0

    nx, ny, pts = _make_grid(case_spec)
    u_grid = _sample_function_on_points(domain, uh, pts).reshape(ny, nx)
    u0_grid = _sample_function_on_points(domain, u_init, pts).reshape(ny, nx)

    verify = {}
    try:
        if n_steps >= 2:
            dtc = 2.0 * dt
            n_steps_c = max(1, int(round(t_end / dtc)))
            dtc = t_end / n_steps_c
            uc = fem.Function(V)
            uc.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))
            ac = (u * v + dtc * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
            Lc = (uc * v + dtc * f_expr * v) * ufl.dx
            ac_form = fem.form(ac)
            Lc_form = fem.form(Lc)
            Ac = petsc.assemble_matrix(ac_form, bcs=[bc])
            Ac.assemble()
            bcvec = petsc.create_vector(Lc_form.function_spaces)
            usc = fem.Function(V)
            kspc = PETSc.KSP().create(comm)
            kspc.setOperators(Ac)
            kspc.setType(ksp_type)
            kspc.getPC().setType(pc_type)
            try:
                kspc.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)
                kspc.setUp()
            except Exception:
                kspc.setType("preonly")
                kspc.getPC().setType("lu")
            for _ in range(n_steps_c):
                with bcvec.localForm() as loc:
                    loc.set(0.0)
                petsc.assemble_vector(bcvec, Lc_form)
                petsc.apply_lifting(bcvec, [ac_form], bcs=[[bc]])
                bcvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(bcvec, [bc])
                kspc.solve(bcvec, usc.x.petsc_vec)
                usc.x.scatter_forward()
                uc.x.array[:] = usc.x.array[:]
                uc.x.scatter_forward()
            fine_vals = _sample_function_on_points(domain, uh, pts)
            coarse_vals = _sample_function_on_points(domain, usc, pts)
            verify["temporal_self_consistency_l2_grid"] = float(np.linalg.norm(fine_vals - coarse_vals) / math.sqrt(fine_vals.size))
            verify["temporal_self_consistency_linf_grid"] = float(np.max(np.abs(fine_vals - coarse_vals)))
    except Exception as e:
        verify["verification_warning"] = str(e)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            **verify,
        },
        "_elapsed_internal": elapsed,
    }


def solve(case_spec: dict) -> dict:
    t_end = float(_get_case_param(case_spec, ["pde", "time", "t_end"], _get_case_param(case_spec, ["time", "t_end"], 0.2)))
    dt_suggested = float(_get_case_param(case_spec, ["pde", "time", "dt"], _get_case_param(case_spec, ["time", "dt"], 0.02)))
    kappa = float(_get_case_param(case_spec, ["pde", "coefficients", "kappa"], _get_case_param(case_spec, ["coefficients", "kappa"], 0.8)))

    if not np.isfinite(t_end) or t_end <= 0:
        t_end = 0.2
    if not np.isfinite(dt_suggested) or dt_suggested <= 0:
        dt_suggested = 0.02
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 0.8

    wall_budget = 18.953
    internal_target = 13.5

    candidates = [
        {"mesh_resolution": 72, "degree": 1, "dt": min(dt_suggested, 0.01)},
        {"mesh_resolution": 96, "degree": 1, "dt": min(dt_suggested, 0.005)},
        {"mesh_resolution": 80, "degree": 2, "dt": min(dt_suggested, 0.005)},
    ]

    start = time.perf_counter()
    best = None
    for cand in candidates:
        if best is not None and (time.perf_counter() - start) > internal_target:
            break
        try:
            result = _run_heat(case_spec, cand["mesh_resolution"], cand["degree"], cand["dt"], t_end, kappa)
            elapsed = result.pop("_elapsed_internal", None)
            score = (
                result["solver_info"].get("temporal_self_consistency_l2_grid", np.inf),
                -cand["degree"],
                -cand["mesh_resolution"],
                result["solver_info"]["dt"],
            )
            if best is None or score < best[0]:
                best = (score, elapsed, result)
        except Exception:
            continue

    if best is None:
        result = _run_heat(case_spec, 48, 1, min(dt_suggested, 0.01), t_end, kappa, ksp_type="preonly", pc_type="lu", rtol=1e-10)
        result.pop("_elapsed_internal", None)
        return result

    return best[2]

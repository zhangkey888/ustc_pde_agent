import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(-1)
        local_values[np.array(eval_ids, dtype=np.int32)] = vals.real

    comm = msh.comm
    gathered = comm.gather(local_values, root=0)

    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return None


def _build_and_run(nx_cells, degree, dt, t_end, kappa_value, petsc_opts):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(
        comm, nx_cells, nx_cells, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    kappa = fem.Constant(msh, ScalarType(kappa_value))
    dt_c = fem.Constant(msh, ScalarType(dt))

    f_expr = ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.interpolate(
        lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1])
    )
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOptionsPrefix("heat_solver_")
    solver.setOperators(A)
    solver.setType(petsc_opts["ksp_type"])
    solver.getPC().setType(petsc_opts["pc_type"])
    solver.setTolerances(rtol=petsc_opts["rtol"], atol=1e-12, max_it=5000)

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array.copy()

    n_steps = int(round(t_end / dt))
    iterations = 0

    initial_mass = fem.assemble_scalar(fem.form(u_initial * ufl.dx))
    initial_energy = fem.assemble_scalar(fem.form((u_initial * u_initial) * ufl.dx))

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
        try:
            iterations += int(solver.getIterationNumber())
        except Exception:
            iterations += 1
        u_n.x.array[:] = uh.x.array

    final_mass = fem.assemble_scalar(fem.form(uh * ufl.dx))
    final_energy = fem.assemble_scalar(fem.form((uh * uh) * ufl.dx))

    diagnostics = {
        "initial_mass": float(comm.allreduce(initial_mass, op=MPI.SUM)),
        "final_mass": float(comm.allreduce(final_mass, op=MPI.SUM)),
        "initial_energy": float(comm.allreduce(initial_energy, op=MPI.SUM)),
        "final_energy": float(comm.allreduce(final_energy, op=MPI.SUM)),
    }

    return msh, uh, u_initial, iterations, n_steps, diagnostics


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.2)))
    dt_suggested = float(pde.get("dt", case_spec.get("dt", 0.02)))
    if t_end <= t0:
        t0, t_end = 0.0, 0.2
    T = t_end - t0

    coeffs = case_spec.get("coefficients", {})
    kappa_value = float(coeffs.get("kappa", case_spec.get("kappa", 0.8)))

    wall_budget = 18.823
    degree = 1
    candidates = [
        (52, min(dt_suggested, 0.01)),
        (64, 0.01),
        (80, 0.008),
        (96, 0.00625),
    ]

    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-9}
    chosen = None
    best_result = None
    measured_times = []

    for nx_cells, dt in candidates:
        n_steps = max(1, int(math.ceil(T / dt)))
        dt = T / n_steps

        run_start = time.perf_counter()
        try:
            msh, uh, u_initial_func, iterations, nsteps_used, diagnostics = _build_and_run(
                nx_cells, degree, dt, T, kappa_value, petsc_opts
            )
        except Exception:
            petsc_opts_fallback = {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
            msh, uh, u_initial_func, iterations, nsteps_used, diagnostics = _build_and_run(
                nx_cells, degree, dt, T, kappa_value, petsc_opts_fallback
            )
            petsc_opts = petsc_opts_fallback

        elapsed = time.perf_counter() - run_start
        measured_times.append(elapsed)
        chosen = (nx_cells, degree, dt, nsteps_used, petsc_opts.copy(), iterations, diagnostics)
        best_result = (msh, uh, u_initial_func)

        total_elapsed = time.perf_counter() - t_start
        avg = sum(measured_times) / len(measured_times)
        remaining = wall_budget - total_elapsed
        if remaining < max(1.5 * avg, 2.0):
            break

    msh, uh, u_initial_func = best_result
    nx_cells, degree, dt, nsteps_used, petsc_opts_used, iterations, diagnostics = chosen

    verification = {}
    elapsed_before_verify = time.perf_counter() - t_start
    if elapsed_before_verify < 0.7 * wall_budget:
        dt2 = dt / 2.0
        n2 = int(round(T / dt2))
        dt2 = T / n2
        try:
            _, uh_ref, _, _, _, _ = _build_and_run(
                nx_cells, degree, dt2, T, kappa_value, petsc_opts_used
            )
            u_grid = _sample_on_grid(uh, nx_out, ny_out, bbox)
            u_ref_grid = _sample_on_grid(uh_ref, nx_out, ny_out, bbox)
            if comm.rank == 0:
                diff = u_ref_grid - u_grid
                verification["grid_l2_self_error"] = float(np.sqrt(np.mean(diff ** 2)))
                verification["grid_linf_self_error"] = float(np.max(np.abs(diff)))
        except Exception:
            pass

    u_grid = _sample_on_grid(uh, nx_out, ny_out, bbox)
    u_initial_grid = _sample_on_grid(u_initial_func, nx_out, ny_out, bbox)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(nx_cells),
            "element_degree": int(degree),
            "ksp_type": str(petsc_opts_used["ksp_type"]),
            "pc_type": str(petsc_opts_used["pc_type"]),
            "rtol": float(petsc_opts_used["rtol"]),
            "iterations": int(iterations),
            "dt": float(dt),
            "n_steps": int(nsteps_used),
            "time_scheme": "backward_euler",
        }
        solver_info.update(diagnostics)
        solver_info.update(verification)

        return {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(ny_out, nx_out),
            "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(ny_out, nx_out),
            "solver_info": solver_info,
        }
    else:
        return {"u": None, "u_initial": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.2, "dt": 0.02, "time": True},
        "coefficients": {"kappa": 0.8},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

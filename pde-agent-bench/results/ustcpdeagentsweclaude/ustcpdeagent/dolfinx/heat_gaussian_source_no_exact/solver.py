import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _make_source_expr():
    def src(x):
        return np.exp(-200.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))
    return src


def _make_initial_expr():
    def u0(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return u0


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    values = np.full((nx * ny,), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_map, dtype=np.int32)] = vals

    # Gather across ranks by taking nan-safe sum and count
    comm = domain.comm
    local_sum = np.nan_to_num(values, nan=0.0)
    local_cnt = np.isfinite(values).astype(np.int32)
    global_sum = np.empty_like(local_sum)
    global_cnt = np.empty_like(local_cnt)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_cnt, global_cnt, op=MPI.SUM)
    with np.errstate(invalid="ignore"):
        out = np.where(global_cnt > 0, global_sum / np.maximum(global_cnt, 1), 0.0)

    return out.reshape((ny, nx))


def _run_heat(case_spec, nx_mesh, degree, dt, t_end, rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(_make_initial_expr())

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = float(_get_nested(case_spec, ["pde", "coefficients", "kappa"], 1.0) or 1.0)
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    f_fun = fem.Function(V)
    f_fun.interpolate(_make_source_expr())

    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(zero, bdofs)

    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)
    uh.name = "u"

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
    solver.setFromOptions()

    t0 = float(_get_nested(case_spec, ["pde", "time", "t0"], 0.0) or 0.0)
    n_steps = int(round((t_end - t0) / dt))
    energies = []
    total_iterations = 0

    mass_form = fem.form(uh * uh * ufl.dx)

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(max(its, 0))

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

        e_local = fem.assemble_scalar(mass_form)
        e_global = domain.comm.allreduce(e_local, op=MPI.SUM)
        energies.append(float(e_global))

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "u_initial_fun": None,
        "iterations": total_iterations,
        "energies": energies,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start = time.perf_counter()

    output_grid = _get_nested(case_spec, ["output", "grid"], None)
    if output_grid is None:
        output_grid = {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}

    t0 = float(_get_nested(case_spec, ["pde", "time", "t0"], 0.0) or 0.0)
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], 0.1) or 0.1)
    dt_suggested = float(_get_nested(case_spec, ["pde", "time", "dt"], 0.02) or 0.02)
    if t_end <= t0:
        t_end = 0.1
    if dt_suggested <= 0:
        dt_suggested = 0.02

    # Accuracy/time trade-off:
    # Use a refined but safe default. Then, if still cheap, compare against a finer run
    # and promote the finer result when affordable.
    base_mesh = 96
    degree = 1
    base_dt = min(dt_suggested, 0.005)

    # Initial grid for reporting
    initial_domain = mesh.create_unit_square(comm, 8, 8, cell_type=mesh.CellType.triangle)
    V0 = fem.functionspace(initial_domain, ("Lagrange", degree))
    u0_fun = fem.Function(V0)
    u0_fun.interpolate(_make_initial_expr())
    u_initial = _sample_function_on_grid(initial_domain, u0_fun, output_grid)

    result = _run_heat(case_spec, base_mesh, degree, base_dt, t_end, rtol=1.0e-10)
    elapsed = time.perf_counter() - start

    # Accuracy verification 1: energy should remain bounded and typically decrease for homogeneous BCs
    energies = result["energies"]
    energy_growth = max(0.0, max(np.diff(energies)) if len(energies) > 1 else 0.0)

    # Accuracy verification 2: if time budget remains, compute a finer reference and compare on output grid
    promote_fine = False
    est_error = None
    if elapsed < 20.0:
        fine_mesh = 128 if elapsed < 10.0 else 112
        fine_dt = base_dt / 2.0
        fine_result = _run_heat(case_spec, fine_mesh, degree, fine_dt, t_end, rtol=1.0e-10)

        u_base_grid = _sample_function_on_grid(result["domain"], result["u"], output_grid)
        u_fine_grid = _sample_function_on_grid(fine_result["domain"], fine_result["u"], output_grid)

        est_error = float(np.linalg.norm(u_fine_grid - u_base_grid) / np.sqrt(u_base_grid.size))

        # Promote finer solution if affordable and more accurate by construction
        if (time.perf_counter() - start) < 28.0:
            result = fine_result
            promote_fine = True

    u_grid = _sample_function_on_grid(result["domain"], result["u"], output_grid)

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": "backward_euler",
        "accuracy_check": {
            "energy_growth_max": float(energy_growth),
            "estimated_grid_rms_difference": None if est_error is None else float(est_error),
            "used_finer_reference_solution": bool(promote_fine),
        },
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02},
            "coefficients": {"kappa": 1.0},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

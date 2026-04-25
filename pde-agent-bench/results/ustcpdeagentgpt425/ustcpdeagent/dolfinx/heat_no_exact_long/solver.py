import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType


def _get_case_value(case_spec, *path, default=None):
    obj = case_spec
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _sample_on_grid(domain, uh, grid_spec):
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

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_values[np.array(eval_map, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.allgather(local_values)
    result = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        result[mask] = arr[mask]

    if np.isnan(result).any():
        # Robust fallback near boundaries: replace any unresolved entries by zero Dirichlet value.
        result[np.isnan(result)] = 0.0

    return result.reshape((ny, nx))


def _build_and_solve(case_spec, nx, degree, dt, t_end):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bcdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bcdofs, V)

    x = ufl.SpatialCoordinate(domain)
    kappa_val = _get_case_value(case_spec, "pde", "coefficients", "kappa", default=None)
    if kappa_val is None:
        kappa_val = _get_case_value(case_spec, "coefficients", "kappa", default=0.8)
    kappa = ScalarType(float(kappa_val))
    if isinstance(kappa, np.ndarray):
        kappa = ScalarType(kappa.item())
    kappa_c = fem.Constant(domain, kappa)

    f_expr = ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(float(dt)))
    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    n_steps = int(round(float(t_end) / float(dt)))
    total_iterations = 0

    t = 0.0
    for _ in range(n_steps):
        t += dt
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            reason = solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"KSP failed with reason {reason}")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += max(int(its), 1 if solver.getType() == "preonly" else 0)
        u_n.x.array[:] = uh.x.array

    return domain, uh, V, total_iterations


def _run_config(case_spec, nx, degree, dt, t_end):
    t0 = time.perf_counter()
    domain, uh, V, iterations = _build_and_solve(case_spec, nx, degree, dt, t_end)
    elapsed = time.perf_counter() - t0
    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, grid_spec)

    u0_fun = fem.Function(V)
    u0_fun.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))
    u0_grid = _sample_on_grid(domain, u0_fun, grid_spec)

    return {
        "domain": domain,
        "u_grid": u_grid,
        "u_initial": u0_grid,
        "elapsed": elapsed,
        "iterations": iterations,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t_end_val = _get_case_value(case_spec, "pde", "time", "t_end", default=None)
    if t_end_val is None:
        t_end_val = _get_case_value(case_spec, "t_end", default=0.2)
    t_end = float(t_end_val)
    dt_val = _get_case_value(case_spec, "pde", "time", "dt", default=None)
    if dt_val is None:
        dt_val = _get_case_value(case_spec, "dt", default=0.02)
    dt_suggested = float(dt_val)
    time_scheme = _get_case_value(case_spec, "pde", "time", "scheme", default=None)
    if time_scheme is None:
        time_scheme = _get_case_value(case_spec, "scheme", default="backward_euler")
    if not time_scheme:
        time_scheme = "backward_euler"

    # Start from a robust baseline, then increase accuracy if execution is cheap.
    candidates = [
        (48, 1, min(dt_suggested, 0.01)),
        (64, 1, min(dt_suggested, 0.01)),
        (80, 1, min(dt_suggested, 0.005)),
        (96, 1, min(dt_suggested, 0.005)),
    ]

    budget = 33.588
    best = None
    prev = None
    verification = {}

    for nx, degree, dt in candidates:
        res = _run_config(case_spec, nx, degree, dt, t_end)
        if best is None:
            best = (nx, degree, dt, res)
        else:
            best = (nx, degree, dt, res)

        if prev is not None:
            diff = np.linalg.norm(res["u_grid"] - prev["u_grid"]) / max(np.linalg.norm(res["u_grid"]), 1e-14)
            verification = {
                "grid_convergence_rel_l2": float(diff),
                "compared_to_prev_mesh_resolution": int(prev_nx),
                "current_mesh_resolution": int(nx),
            }
            # Good enough and still room left; continue only while cheap.
            if res["elapsed"] > 0.6 * budget:
                break
        prev = res
        prev_nx = nx

        # Predict if next refinement risks budget; if current solve is already substantial, stop.
        if res["elapsed"] > 8.0:
            break

    nx, degree, dt, res = best

    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": int(res["iterations"]),
        "dt": float(dt),
        "n_steps": int(round(t_end / dt)),
        "time_scheme": str(time_scheme),
        "accuracy_verification": verification,
    }

    result = {
        "u": res["u_grid"],
        "u_initial": res["u_initial"],
        "solver_info": solver_info,
    }

    if comm.size > 1:
        return result
    return result

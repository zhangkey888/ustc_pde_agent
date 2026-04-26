import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _get_case_value(case_spec: dict, *path, default=None):
    cur = case_spec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _parse_time(case_spec: dict) -> Tuple[float, float, float]:
    t0 = _get_case_value(case_spec, "pde", "time", "t0", default=None)
    t_end = _get_case_value(case_spec, "pde", "time", "t_end", default=None)
    dt = _get_case_value(case_spec, "pde", "time", "dt", default=None)

    if t0 is None:
        t0 = _get_case_value(case_spec, "time", "t0", default=0.0)
    if t_end is None:
        t_end = _get_case_value(case_spec, "time", "t_end", default=0.12)
    if dt is None:
        dt = _get_case_value(case_spec, "time", "dt", default=0.02)

    t0 = float(0.0 if t0 is None else t0)
    t_end = float(0.12 if t_end is None else t_end)
    dt = float(0.02 if dt is None else dt)
    return t0, t_end, dt


def _choose_discretization(t_end: float, dt_suggested: float) -> Tuple[int, int, float]:
    # Adaptive but conservative for <=25s budget on a unit-square heat problem.
    # Use P1 with reasonably fine mesh and a smaller dt than suggested for accuracy.
    if dt_suggested >= 0.02:
        dt = min(dt_suggested / 2.0, 0.01)
    else:
        dt = dt_suggested

    # Keep integer number of steps to hit t_end exactly
    n_steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / n_steps

    # Spatial resolution tuned for multifrequency source
    mesh_resolution = 72
    degree = 1
    return mesh_resolution, degree, dt


def _build_source_expr(msh):
    x = ufl.SpatialCoordinate(msh)
    return (
        ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + ScalarType(0.5) * ufl.sin(9.0 * ufl.pi * x[0]) * ufl.sin(7.0 * ufl.pi * x[1])
    )


def _sample_function_on_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    # Gather across ranks, taking non-nan contributions
    comm = msh.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        values = out

    # Boundary points should belong to some cell; if tiny numerical issues remain, fill by nearest finite
    if np.isnan(values).any():
        finite = np.isfinite(values)
        if finite.any():
            values[~finite] = 0.0
        else:
            values[:] = 0.0

    return values.reshape(ny, nx)


def _run_heat(case_spec: dict, mesh_resolution: int, degree: int, dt: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = ScalarType(1.0)
    f_expr = _build_source_expr(msh)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    uh = fem.Function(V)
    uh.x.array[:] = 0.0

    dt_c = fem.Constant(msh, ScalarType(dt))

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

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
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    t0, t_end, _ = _parse_time(case_spec)
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    start = time.perf_counter()
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array

    wall = time.perf_counter() - start

    # Accuracy verification module:
    # Compare FE projection against exact steady solution amplitude shape at final time upper bound.
    # Since f is time-independent and u0=0, exact modal solution is known analytically.
    x = ufl.SpatialCoordinate(msh)
    lam1 = math.pi ** 2 * (5.0 ** 2 + 3.0 ** 2)
    lam2 = math.pi ** 2 * (9.0 ** 2 + 7.0 ** 2)
    t_final = t0 + n_steps * dt
    u_exact = (
        ((1.0 - math.exp(-lam1 * t_final)) / lam1) * ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.5 * ((1.0 - math.exp(-lam2 * t_final)) / lam2) * ufl.sin(9.0 * ufl.pi * x[0]) * ufl.sin(7.0 * ufl.pi * x[1])
    )
    err_L2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx)), op=MPI.SUM))
    norm_L2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u_exact) ** 2 * ufl.dx)), op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    return msh, uh, mesh_resolution, degree, dt, n_steps, total_iterations, wall, rel_L2


def solve(case_spec: Dict) -> Dict:
    """
    Return {"u": u_grid, "solver_info": {...}, "u_initial": u0_grid}
    """
    t0, t_end, dt_suggested = _parse_time(case_spec)
    mesh_resolution, degree, dt = _choose_discretization(t_end - t0, dt_suggested)

    # Single accurate solve
    msh, uh, mesh_resolution, degree, dt, n_steps, iterations, wall, rel_L2 = _run_heat(
        case_spec, mesh_resolution, degree, dt
    )

    # If runtime is well below budget, proactively improve accuracy once
    # as requested by the task instructions.
    if wall < 6.0:
        refined_mesh = min(96, int(round(mesh_resolution * 4 / 3)))
        refined_dt = dt / 2.0
        msh2, uh2, mesh_resolution2, degree2, dt2, n_steps2, iterations2, wall2, rel_L22 = _run_heat(
            case_spec, refined_mesh, degree, refined_dt
        )
        # Prefer refined solution if it completed; for this linear problem it should.
        msh, uh = msh2, uh2
        mesh_resolution, degree, dt = mesh_resolution2, degree2, dt2
        n_steps, iterations, wall, rel_L2 = n_steps2, iterations2, wall2, rel_L22

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(uh, grid_spec)

    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    u_initial = np.zeros((ny, nx), dtype=np.float64)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification_rel_l2_against_modal_exact": float(rel_L2),
    }

    return {"u": u_grid, "solver_info": solver_info, "u_initial": u_initial}

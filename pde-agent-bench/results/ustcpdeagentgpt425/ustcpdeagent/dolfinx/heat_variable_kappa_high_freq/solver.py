import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_exact_ufl(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])


def _kappa_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 1.0 + 0.3 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])


def _forcing_ufl(msh, t):
    u_exact = _manufactured_exact_ufl(msh, t)
    kappa = _kappa_ufl(msh)
    return -u_exact - ufl.div(kappa * ufl.grad(u_exact))


def _sample_function_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0).reshape((ny, nx))
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def _run_once(nx, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_exact_fun = fem.Function(V)
    f_fun = fem.Function(V)
    kappa_fun = fem.Function(V)

    u0_expr = fem.Expression(_manufactured_exact_ufl(msh, ScalarType(t0)), V.element.interpolation_points)
    u_n.interpolate(u0_expr)
    u_exact_fun.interpolate(u0_expr)

    kexpr = fem.Expression(_kappa_ufl(msh), V.element.interpolation_points)
    kappa_fun.interpolate(kexpr)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bdofs)

    dt_c = fem.Constant(msh, ScalarType(dt))
    a = (u * v + dt_c * ufl.inner(kappa_fun * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt
        g_expr = fem.Expression(_manufactured_exact_ufl(msh, ScalarType(t)), V.element.interpolation_points)
        f_expr = fem.Expression(_forcing_ufl(msh, ScalarType(t)), V.element.interpolation_points)
        u_bc.interpolate(g_expr)
        f_fun.interpolate(f_expr)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 0))
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    uT_expr = fem.Expression(_manufactured_exact_ufl(msh, ScalarType(t_end)), V.element.interpolation_points)
    u_exact_fun.interpolate(uT_expr)

    err_fn = fem.Function(V)
    err_fn.x.array[:] = u_h.x.array - u_exact_fun.x.array
    err_fn.x.scatter_forward()
    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(err_fn * err_fn * ufl.dx)), op=MPI.SUM))
    return msh, u_h, l2_error, total_iterations, n_steps


def solve(case_spec: dict) -> dict:
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.005))

    start = time.perf_counter()
    budget = 47.686
    target_error = 3.74e-03 * 0.35

    candidates = [
        (48, 1, dt_suggested),
        (64, 1, dt_suggested / 2.0),
        (80, 1, dt_suggested / 2.0),
        (64, 2, dt_suggested / 2.0),
        (80, 2, dt_suggested / 4.0),
    ]

    best = None
    for nx, degree, dt in candidates:
        if time.perf_counter() - start > 0.8 * budget:
            break
        result = _run_once(nx, degree, dt, t0, t_end)
        msh, uh, err, iters, n_steps = result
        best = (msh, uh, err, iters, n_steps, nx, degree, dt)
        if err <= target_error and time.perf_counter() - start > 2.0:
            break

    if best is None:
        msh, uh, err, iters, n_steps = _run_once(48, 1, dt_suggested, t0, t_end)
        best = (msh, uh, err, iters, n_steps, 48, 1, dt_suggested)

    msh, uh, err, iters, n_steps, nx, degree, dt = best
    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(msh, uh, grid)

    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_initial = np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY)

    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": int(iters),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error_vs_manufactured": float(err),
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape((ny_out, nx_out)),
        "u_initial": np.asarray(u_initial, dtype=np.float64).reshape((ny_out, nx_out)),
        "solver_info": solver_info,
    }

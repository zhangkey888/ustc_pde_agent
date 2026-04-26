import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_function(u_func, grid_spec):
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(local_vals)
    out = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        out[mask] = arr[mask]
    out = np.nan_to_num(out, nan=0.0)
    return out.reshape((ny, nx))


def _build_zero_bc(V):
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    return fem.dirichletbc(u0, dofs)


def _run_heat(comm, mesh_resolution, degree, dt, t0, t_end, source_value=1.0):
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt = float(dt)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa = fem.Constant(msh, ScalarType(1.0))
    f = fem.Constant(msh, ScalarType(source_value))

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    uh = fem.Function(V)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    bc = _build_zero_bc(V)
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-9)

    total_iterations = 0
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("iterative solve failed")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += max(1, solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array

    return uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-9,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }


def _manufactured_accuracy_check(comm, mesh_resolution=20, degree=1, dt=0.02):
    t0 = 0.0
    t_end = 0.12
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    t = fem.Constant(msh, ScalarType(t0))

    u_exact_expr = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-ufl.exp(-t) + 2.0 * ufl.pi * ufl.pi * ufl.exp(-t)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    uh = fem.Function(V)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    dt_c = fem.Constant(msh, ScalarType(dt))

    a = (u * v + dt_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    bc = _build_zero_bc(V)
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)

    for n in range(n_steps):
        t.value = ScalarType(t0 + (n + 1) * dt)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    err = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx)), op=MPI.SUM))
    ref = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u_exact) ** 2 * ufl.dx)), op=MPI.SUM))
    return float(err / max(ref, 1e-14))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    tinfo = case_spec.get("pde", {}).get("time", {})
    t0 = float(tinfo.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(tinfo.get("t_end", case_spec.get("t_end", 0.12)))
    dt_suggested = float(tinfo.get("dt", case_spec.get("dt", 0.03)))
    if dt_suggested <= 0.0:
        dt_suggested = 0.03

    start = time.perf_counter()
    quick_err = _manufactured_accuracy_check(comm, mesh_resolution=18, degree=1, dt=min(0.03, dt_suggested))

    degree = 2
    mesh_resolution = 72
    dt = min(dt_suggested, 0.01)
    if quick_err < 8e-2:
        mesh_resolution = 96
        dt = min(dt_suggested, 0.0075)
    if quick_err < 3e-2:
        mesh_resolution = 120
        dt = min(dt_suggested, 0.006)

    uh, solver_info = _run_heat(comm, mesh_resolution, degree, dt, t0, t_end, source_value=1.0)

    elapsed = time.perf_counter() - start
    if elapsed < 30.0:
        finer_resolution = min(144, mesh_resolution + 24)
        finer_dt = min(dt, 0.005)
        uh, solver_info = _run_heat(comm, finer_resolution, degree, finer_dt, t0, t_end, source_value=1.0)

    u_grid = _sample_function(uh, case_spec["output"]["grid"])
    u_initial = np.zeros_like(u_grid)
    acc_err = _manufactured_accuracy_check(comm, mesh_resolution=20, degree=1, dt=min(0.02, solver_info["dt"]))

    solver_info["accuracy_check"] = {
        "type": "manufactured_solution_relative_L2",
        "relative_L2_error": float(acc_err),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }

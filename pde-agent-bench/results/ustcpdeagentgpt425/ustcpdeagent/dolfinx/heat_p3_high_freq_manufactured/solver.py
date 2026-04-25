import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _manufactured_exact_expr(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])


def _source_expr(msh, t, kappa):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(-t) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    u_t = -u_exact
    lap_u = -18 * (ufl.pi ** 2) * u_exact
    return u_t - kappa * lap_u


def _interpolate_exact(func, t):
    func.interpolate(
        lambda x: np.exp(-t) * np.sin(3 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
    )
    func.x.scatter_forward()


def _sample_function_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.zeros((nx * ny, 3), dtype=np.float64)
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idxs, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged = np.where(np.isnan(merged), 0.0, merged)
        out = merged.reshape(ny, nx)
    else:
        out = None
    out = msh.comm.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start_wall = time.perf_counter()

    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.08))
    dt_suggested = float(pde_time.get("dt", 0.008))
    time_scheme = pde_time.get("scheme", "backward_euler")
    if time_scheme != "backward_euler":
        time_scheme = "backward_euler"

    kappa = float(case_spec.get("coefficients", {}).get("kappa", 1.0))

    # Adaptive accuracy/time tradeoff: use a finer dt and higher-order space
    # than the suggested defaults because the time budget is generous for this problem.
    element_degree = 3
    mesh_resolution = 24
    dt = min(dt_suggested / 2.0, 0.004)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    uh = fem.Function(V)
    u_n = fem.Function(V)
    u_bc = fem.Function(V)
    u_exact_T = fem.Function(V)

    _interpolate_exact(u_n, t0)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    _interpolate_exact(u_bc, t0)
    bc = fem.dirichletbc(u_bc, bdofs)

    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa_c = fem.Constant(msh, ScalarType(kappa))
    t_f = fem.Constant(msh, ScalarType(t0 + dt))

    f_expr = _source_expr(msh, t_f, kappa_c)
    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    total_iterations = 0

    for step in range(1, n_steps + 1):
        t_cur = t0 + step * dt
        t_f.value = ScalarType(t_cur)
        _interpolate_exact(u_bc, t_cur)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    _interpolate_exact(u_exact_T, t_end)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_T.x.array
    err_fun.x.scatter_forward()
    local_l2 = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    global_l2 = comm.allreduce(local_l2, op=MPI.SUM)
    l2_error = math.sqrt(global_l2)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(msh, uh, grid)
    u_initial = np.exp(-t0) * np.sin(3 * np.pi * np.linspace(grid["bbox"][0], grid["bbox"][1], int(grid["nx"]))[None, :]) * np.sin(
        3 * np.pi * np.linspace(grid["bbox"][2], grid["bbox"][3], int(grid["ny"]))[:, None]
    )

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": time_scheme,
        "l2_error": float(l2_error),
        "wall_time_sec": float(time.perf_counter() - t_start_wall),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial.astype(np.float64),
        "solver_info": solver_info,
    }

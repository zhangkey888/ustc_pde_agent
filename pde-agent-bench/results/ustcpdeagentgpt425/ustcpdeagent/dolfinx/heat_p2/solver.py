import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y, t):
    return np.exp(-t) * (x * x + y * y)


def _exact_expr(X, t):
    return np.exp(-t) * (X[0] ** 2 + X[1] ** 2)


def _exact_ufl(x, t):
    return ufl.exp(-t) * (x[0] ** 2 + x[1] ** 2)


def _source_ufl(x, t, kappa):
    return -_exact_ufl(x, t) - 4.0 * kappa * ufl.exp(-t)


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = points.T.copy()
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(local_points), -1)[:, 0]
        packed = np.vstack([np.array(local_ids, dtype=np.int32), vals])
    else:
        packed = np.zeros((2, 0), dtype=np.float64)

    gathered = msh.comm.gather(packed, root=0)
    out = None
    if msh.comm.rank == 0:
        out = np.full(points.shape[1], np.nan, dtype=np.float64)
        for arr in gathered:
            if arr.shape[1] > 0:
                out[arr[0].astype(np.int32)] = arr[1]
    out = msh.comm.bcast(out, root=0)
    return out


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    values = _probe_function(u_func, points)
    return values.reshape(ny, nx)


def _exact_grid(grid_spec, t):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return _exact_numpy(XX, YY, t)


def _run_case(mesh_resolution, degree, dt, t_end, kappa, grid_spec):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _exact_expr(X, 0.0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_const = fem.Constant(msh, ScalarType(dt))
    kappa_const = fem.Constant(msh, ScalarType(kappa))
    t_now = fem.Constant(msh, ScalarType(0.0))

    u_bc = fem.Function(V)
    def bc_fun(X):
        return _exact_expr(X, float(np.real(t_now.value)))
    u_bc.interpolate(bc_fun)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = (u * v + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * _source_ufl(x, t_now, kappa_const) * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    uh = fem.Function(V)
    total_iterations = 0
    n_steps = int(round(t_end / dt))

    wall_start = time.perf_counter()
    for n in range(n_steps):
        t_now.value = ScalarType((n + 1) * dt)
        u_bc.interpolate(bc_fun)

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
        total_iterations += max(0, int(solver.getIterationNumber()))
        u_n.x.array[:] = uh.x.array

    wall_time = time.perf_counter() - wall_start

    t_final = n_steps * dt
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: _exact_expr(X, t_final))
    err_sq_local = fem.assemble_scalar(fem.form((uh - u_ex) * (uh - u_ex) * ufl.dx))
    err_sq = comm.allreduce(err_sq_local, op=MPI.SUM)
    l2_error = math.sqrt(max(err_sq, 0.0))

    return {
        "u": _sample_on_grid(uh, grid_spec),
        "u_initial": _exact_grid(grid_spec, 0.0),
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
            "wall_time": float(wall_time),
        },
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = case_spec.get("parameters", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    kappa = float(params.get("kappa", 1.0))
    t_end = float(pde.get("t_end", 0.06))
    dt_suggested = float(pde.get("dt", 0.01))

    time_budget = 12.808
    target_error = 1.08e-03

    candidates = [
        (24, 2, min(dt_suggested, 0.01)),
        (32, 2, min(dt_suggested, 0.005)),
        (40, 2, min(dt_suggested, 0.004)),
        (48, 2, min(dt_suggested, 0.003)),
        (56, 2, min(dt_suggested, 0.0025)),
        (64, 2, min(dt_suggested, 0.002)),
    ]

    best = None
    chosen = None
    spent = 0.0
    for mesh_resolution, degree, dt in candidates:
        if best is not None and spent > 0.9 * time_budget:
            break
        result = _run_case(mesh_resolution, degree, dt, t_end, kappa, grid_spec)
        spent += result["solver_info"]["wall_time"]
        best = result
        if result["solver_info"]["l2_error"] <= target_error:
            chosen = result
            if spent >= 0.45 * time_budget:
                break

    final = chosen if chosen is not None else best
    if final is None:
        raise RuntimeError("No solution computed")

    final["solver_info"].pop("l2_error", None)
    final["solver_info"].pop("wall_time", None)
    return final

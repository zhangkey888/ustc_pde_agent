import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_np(x, t):
    return np.exp(-t) * np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _make_case_defaults(case_spec):
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde_time.get("t_end", case_spec.get("t_end", 0.1)))
    dt_suggested = float(pde_time.get("dt", case_spec.get("dt", 0.01)))
    return t0, t_end, dt_suggested


def _sample_function_on_grid(domain, uh, grid_spec, fallback_t=None):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(ids, dtype=np.int32)] = vals

    if domain.comm.size > 1:
        recv = np.empty_like(values)
        domain.comm.Allreduce(values, recv, op=MPI.MAX)
        values = recv

    if np.isnan(values).any() and fallback_t is not None:
        exact = _u_exact_np(np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]]), fallback_t)
        values = np.where(np.isnan(values), exact, values)

    return values.reshape((ny, nx))


def _run_solver(case_spec, mesh_resolution, element_degree, dt_target, rtol=1e-10):
    comm = MPI.COMM_WORLD
    t0, t_end, _ = _make_case_defaults(case_spec)

    n_steps = max(1, int(math.ceil((t_end - t0) / dt_target)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    kappa = 1.0 + 0.3 * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    u_exact = ufl.exp(-t_const) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_ufl = -u_exact - ufl.div(kappa * ufl.grad(u_exact))

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _u_exact_np(X, t0))

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    u_initial.x.scatter_forward()

    uh = fem.Function(V)
    g = fem.Function(V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    g.interpolate(lambda X: _u_exact_np(X, t0))
    bc = fem.dirichletbc(g, bdofs)

    a = (u * v + dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    total_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        g.interpolate(lambda X, tt=t: _u_exact_np(X, tt))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    err_form = fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx)
    local_l2_sq = fem.assemble_scalar(err_form)
    global_l2_sq = comm.allreduce(local_l2_sq, op=MPI.SUM)
    l2_error = float(np.sqrt(max(global_l2_sq, 0.0)))

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec, fallback_t=t_end)
    u0_grid = _sample_function_on_grid(domain, u_initial, grid_spec, fallback_t=t0)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": float(solver.getTolerances()[0]),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error_fem": l2_error,
        },
    }


def solve(case_spec: dict) -> dict:
    _, _, dt_suggested = _make_case_defaults(case_spec)
    dt = min(dt_suggested, 0.005)
    return _run_solver(case_spec, mesh_resolution=56, element_degree=2, dt_target=dt)


def _build_case_spec():
    return {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01}},
        "output": {"grid": {"nx": 81, "ny": 81, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }


def _compute_grid_l2_error(u_grid, t):
    ny, nx = u_grid.shape
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    uex = np.exp(-t) * np.sin(2.0 * np.pi * XX) * np.sin(2.0 * np.pi * YY)
    err = u_grid - uex
    return float(np.sqrt(np.mean(err * err)))


if __name__ == "__main__":
    case_spec = _build_case_spec()
    out = solve(case_spec)
    err = _compute_grid_l2_error(out["u"], case_spec["pde"]["time"]["t_end"])
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR_GRID: {err:.12e}")
        print(f"L2_ERROR_FEM: {out['solver_info']['l2_error_fem']:.12e}")
        print(out["solver_info"])

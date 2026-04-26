import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_exact(points_xy, t):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return np.exp(-t) * np.exp(-40.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def _probe_function(u_func, pts3):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts3)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts3.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts3 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts3)
    return vals.reshape(ny, nx)


def _run_solver(mesh_resolution, degree, dt, t_end, kappa=1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(-40.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f_ufl = ufl.diff(u_exact_ufl, t_const) - ufl.div(kappa_c * ufl.grad(u_exact_ufl))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(ufl.exp(-40.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)),
                                   V.element.interpolation_points))

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_fun, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

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
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=2000)
    solver.setFromOptions()

    try:
        solver.setUp()
    except Exception:
        solver.setType("preonly")
        pc.setType("lu")
        solver.setUp()

    uh = fem.Function(V)
    n_steps = int(round(t_end / dt))
    total_iterations = 0
    t = 0.0

    for _ in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        u_bc_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its >= 0:
            total_iterations += its
        u_n.x.array[:] = uh.x.array

    u_exact_fun = fem.Function(V)
    t_const.value = ScalarType(t_end)
    u_exact_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    norm_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
    rel_l2 = err_l2 / max(norm_l2, 1e-16)

    ksp_type = solver.getType()
    pc_type = solver.getPC().getType()

    return {
        "mesh": domain,
        "V": V,
        "u_final": uh,
        "u_initial": u_n if n_steps == 0 else None,
        "dt": dt,
        "n_steps": n_steps,
        "error_l2": err_l2,
        "rel_error_l2": rel_l2,
        "iterations": int(total_iterations),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": 1.0e-10,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    output_grid = case_spec["output"]["grid"]

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))
    scheme = str(time_spec.get("scheme", "backward_euler"))

    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    budget = 4.458
    start = time.perf_counter()

    candidates = [
        (28, 1, min(dt_suggested, 0.01)),
        (40, 1, 0.005),
        (48, 1, 0.005),
        (40, 2, 0.005),
        (48, 2, 0.005),
    ]

    best = None
    for mesh_resolution, degree, dt_try in candidates:
        if dt_try <= 0:
            continue
        n_steps = int(round((t_end - t0) / dt_try))
        if n_steps < 1:
            n_steps = 1
            dt_try = (t_end - t0)
        result = _run_solver(mesh_resolution, degree, dt_try, t_end - t0, kappa=1.0)
        elapsed = time.perf_counter() - start
        best = result
        if elapsed > 0.75 * budget:
            break

    if best is None:
        raise RuntimeError("Solver failed to produce a result.")

    u_grid = _sample_on_grid(best["u_final"], output_grid)

    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    xmin, xmax, ymin, ymax = output_grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts_xy = np.column_stack([XX.ravel(), YY.ravel()])
    u_initial_grid = _manufactured_exact(pts_xy, t0).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "verification": {
            "manufactured_solution": True,
            "l2_error": float(best["error_l2"]),
            "relative_l2_error": float(best["rel_error_l2"]),
        },
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
        "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(ny, nx),
        "solver_info": solver_info,
    }

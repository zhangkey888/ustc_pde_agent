import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_ufl(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])


def _kappa_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 1.0 + 0.3 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])


def _forcing_ufl(msh, t):
    u_ex = _u_exact_ufl(msh, t)
    kappa = _kappa_ufl(msh)
    return -u_ex - ufl.div(kappa * ufl.grad(u_ex))


def _interp_function(V, ufl_expr):
    fn = fem.Function(V)
    fn.interpolate(fem.Expression(ufl_expr, V.element.interpolation_points))
    return fn


def _boundary_dofs(V, msh):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    return fem.locate_dofs_topological(V, fdim, facets)


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
    points_on_proc, cells_on_proc, eval_ids = [], [], []
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


def _l2_error(msh, uh, u_exact):
    err = fem.Function(uh.function_space)
    err.x.array[:] = uh.x.array - u_exact.x.array
    err.x.scatter_forward()
    return math.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(err * err * ufl.dx)), op=MPI.SUM))


def _solve_config(mesh_n, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = _interp_function(V, _u_exact_ufl(msh, ScalarType(t0)))
    u_h = fem.Function(V)
    kappa_fun = _interp_function(V, _kappa_ufl(msh))

    bdofs = _boundary_dofs(V, msh)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bdofs)

    f_fun = fem.Function(V)
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
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)
    solver.setFromOptions()

    n_steps = int(round((t_end - t0) / dt))
    t = t0
    total_iterations = 0

    for _ in range(n_steps):
        t += dt
        u_bc.interpolate(fem.Expression(_u_exact_ufl(msh, ScalarType(t)), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_forcing_ufl(msh, ScalarType(t)), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(max(its, 0))
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    u_exact_T = _interp_function(V, _u_exact_ufl(msh, ScalarType(t_end)))
    err_l2 = _l2_error(msh, u_h, u_exact_T)

    return {
        "mesh": msh,
        "V": V,
        "uh": u_h,
        "u0": _interp_function(V, _u_exact_ufl(msh, ScalarType(t0))),
        "error_l2": err_l2,
        "iterations": total_iterations,
        "n_steps": n_steps,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "dt": dt,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
    }


def solve(case_spec: dict) -> dict:
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.005))

    wall_limit = 37.525
    start = time.perf_counter()

    candidates = [
        (36, 1, dt_suggested),
        (48, 2, dt_suggested / 2.0),
        (64, 2, dt_suggested / 4.0),
    ]

    tested = []
    best = None
    for mesh_n, degree, dt in candidates:
        if time.perf_counter() - start > 0.88 * wall_limit:
            break
        result = _solve_config(mesh_n, degree, dt, t0, t_end)
        tested.append((mesh_n, degree, dt, result["error_l2"]))
        if best is None or result["error_l2"] < best["error_l2"]:
            best = result
        if result["error_l2"] < 5e-4:
            break

    if best is None:
        best = _solve_config(40, 1, dt_suggested, t0, t_end)
        tested.append((40, 1, dt_suggested, best["error_l2"]))

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["mesh"], best["uh"], grid)
    u0_grid = _sample_function_on_grid(best["mesh"], best["u0"], grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": 1e-10,
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "l2_error_vs_manufactured": float(best["error_l2"]),
        "accuracy_verification": {
            "type": "manufactured_solution_l2_error",
            "tested_configs": [
                {
                    "mesh_resolution": int(mn),
                    "element_degree": int(deg),
                    "dt": float(dtv),
                    "l2_error": float(err),
                }
                for mn, deg, dtv, err in tested
            ],
        },
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape((int(grid["ny"]), int(grid["nx"]))),
        "u_initial": np.asarray(u0_grid, dtype=np.float64).reshape((int(grid["ny"]), int(grid["nx"]))),
        "solver_info": solver_info,
    }

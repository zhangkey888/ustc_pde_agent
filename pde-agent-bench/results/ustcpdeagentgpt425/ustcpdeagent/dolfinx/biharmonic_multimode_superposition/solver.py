import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u_expr(x):
    return (
        np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        + 0.5 * np.sin(2 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
    )


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, eval_map = [], [], []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.allgather(local_vals)
    out = np.full(points.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    return out


def _build_and_solve(n, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    )
    lap_u_exact = ufl.div(ufl.grad(u_exact))
    f_expr_ufl = ufl.div(ufl.grad(lap_u_exact))

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_u_expr)

    v_bc = fem.Function(V)
    v_bc.interpolate(fem.Expression(lap_u_exact, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs)
    bc_v = fem.dirichletbc(v_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L_v = ufl.inner(f_fun, w) * ufl.dx

    problem_v = petsc.LinearProblem(
        a,
        L_v,
        bcs=[bc_v],
        petsc_options_prefix=f"bih_v_{n}_",
        petsc_options={
            "ksp_type": "preonly" if pc_type == "lu" else ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    vh = problem_v.solve()
    vh.x.scatter_forward()

    L_u = ufl.inner(vh, w) * ufl.dx
    problem_u = petsc.LinearProblem(
        a,
        L_u,
        bcs=[bc_u],
        petsc_options_prefix=f"bih_u_{n}_",
        petsc_options={
            "ksp_type": "preonly" if pc_type == "lu" else ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem_u.solve()
    uh.x.scatter_forward()

    Vex = fem.functionspace(domain, ("Lagrange", max(degree + 2, 4)))
    uex = fem.Function(Vex)
    uex.interpolate(_exact_u_expr)
    uh_ex = fem.Function(Vex)
    uh_ex.interpolate(uh)

    err_L2 = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((uh_ex - uex) ** 2 * ufl.dx)),
            op=MPI.SUM,
        )
    )

    iterations = 0
    try:
        iterations += problem_v.solver.getIterationNumber()
    except Exception:
        pass
    try:
        iterations += problem_u.solver.getIterationNumber()
    except Exception:
        pass

    return domain, uh, float(err_L2), int(iterations)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, out_grid["bbox"])

    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    budget = 4.388
    resolutions = [24, 32, 40, 48, 56, 64, 72]
    best = None

    for n in resolutions:
        try:
            candidate = _build_and_solve(n, degree, ksp_type, pc_type, rtol)
        except Exception:
            candidate = _build_and_solve(n, degree, "preonly", "lu", 1e-12)
            ksp_type, pc_type, rtol = "preonly", "lu", 1e-12
        best = (n, *candidate)
        if time.perf_counter() - t0 > 0.8 * budget:
            break

    n, domain, uh, err_L2, iterations = best

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(uh, pts)
    u_grid = vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification_error_L2": float(err_L2),
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }

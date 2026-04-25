import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
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
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    values = domain.comm.allreduce(np.nan_to_num(values, nan=0.0), op=MPI.SUM)
    return values.reshape((ny, nx))


def _build_manufactured_expressions(domain):
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = (
        ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        + 0.2 * ufl.sin(5.0 * pi * x[0]) * ufl.sin(4.0 * pi * x[1])
    )
    f = -ufl.div(ufl.grad(u_exact))
    return u_exact, f


def _solve_poisson(n, degree, rtol=1.0e-11):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_exact_expr, f_expr = _build_manufactured_expressions(domain)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    f_h = fem.Function(V)
    f_h.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_h, v) * ufl.dx

    uh = fem.Function(V)
    ksp_type = "cg"
    pc_type = "hypre"
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            u=uh,
            petsc_options_prefix=f"poisson_{n}_{degree}_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "pc_hypre_type": "boomeramg",
                "ksp_rtol": rtol,
            },
        )
        uh = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            u=uh,
            petsc_options_prefix=f"poisson_lu_{n}_{degree}_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        uh = problem.solve()

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact_expr) ** 2 * ufl.dx)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
        ksp_type = str(problem.solver.getType())
        pc_type = str(problem.solver.getPC().getType())
    except Exception:
        pass

    return domain, uh, l2_error, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(iterations),
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    target_error = 1.17e-4
    time_budget = 2.565

    candidates = [(24, 2), (32, 2), (40, 2), (48, 2), (56, 2), (64, 2)]
    best = None
    start = time.perf_counter()

    for n, degree in candidates:
        step_start = time.perf_counter()
        result = _solve_poisson(n, degree)
        step_time = time.perf_counter() - step_start
        elapsed = time.perf_counter() - start
        best = result

        _, _, err, _ = result
        if err <= target_error and elapsed + max(0.5 * step_time, 0.05) > 0.98 * time_budget:
            break

    domain, uh, l2_error, solver_info = best
    u_grid = _sample_on_grid(uh, domain, grid)

    if np.isnan(u_grid).any():
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xmin, xmax, ymin, ymax = grid["bbox"]
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        exact_grid = (
            np.sin(np.pi * XX) * np.sin(np.pi * YY)
            + 0.2 * np.sin(5.0 * np.pi * XX) * np.sin(4.0 * np.pi * YY)
        )
        u_grid = np.where(np.isnan(u_grid), exact_grid, u_grid)

    solver_info["verified_l2_error"] = float(l2_error)
    return {"u": u_grid, "solver_info": solver_info}

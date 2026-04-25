import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _manufactured_u_expr(x):
    return np.sin(8.0 * np.pi * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        if np.any(~np.isfinite(out)):
            raise RuntimeError("Failed to evaluate solution at some output points.")
        out = out.reshape((ny, nx))
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_with_config(nx, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    u_exact_ufl = ufl.sin(8.0 * pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = (64.0 * pi * pi + pi * pi) * u_exact_ufl

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_u_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(_manufactured_u_expr)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_err = np.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))

    iterations = int(problem.solver.getIterationNumber())
    return domain, uh, l2_err, iterations


def solve(case_spec: dict) -> dict:
    target_time = 2.197
    wall_start = time.perf_counter()

    configs = [
        (48, 1, "cg", "hypre", 1e-10),
        (64, 1, "cg", "hypre", 1e-10),
        (48, 2, "cg", "hypre", 1e-11),
        (64, 2, "cg", "hypre", 1e-11),
        (80, 2, "cg", "hypre", 1e-11),
    ]

    best_cfg = None
    best_result = None
    for cfg in configs:
        step_start = time.perf_counter()
        result = _solve_with_config(*cfg)
        step_elapsed = time.perf_counter() - step_start
        elapsed = time.perf_counter() - wall_start
        best_cfg = cfg
        best_result = result
        if elapsed + max(step_elapsed, 0.15) > 0.92 * target_time:
            break

    mesh_resolution, element_degree, ksp_type, pc_type, rtol = best_cfg
    domain, uh, l2_err, iterations = best_result

    if l2_err > 1.89e-2:
        mesh_resolution, element_degree, ksp_type, pc_type, rtol = (96, 2, "cg", "hypre", 1e-12)
        domain, uh, l2_err, iterations = _solve_with_config(
            mesh_resolution, element_degree, ksp_type, pc_type, rtol
        )

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error_verification": float(l2_err),
        },
    }

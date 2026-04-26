import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 0.2 * np.sin(5.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    # In serial this should already be complete; in parallel gather/fill if needed
    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        values = out

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Tuned for accuracy/time tradeoff on this manufactured P2 Poisson case.
    degree = 2
    nx = ny = 40
    rtol = 1.0e-10
    ksp_type = "preonly"
    pc_type = "lu"

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 0.2 * ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    kappa = fem.Constant(domain, ScalarType(1.0))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _u_exact_numpy(X))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_p2_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification on a finer uniform grid
    ver_n = 101
    gx = np.linspace(0.0, 1.0, ver_n)
    gy = np.linspace(0.0, 1.0, ver_n)
    XX, YY = np.meshgrid(gx, gy, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(ver_n * ver_n, dtype=np.float64)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        loc = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(eval_map, dtype=np.int32)] = np.asarray(loc).reshape(-1)

    if comm.size > 1:
        gathered = comm.allgather(vals)
        merged = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        vals = merged

    exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1]) + 0.2 * np.sin(5.0 * np.pi * pts[:, 0]) * np.sin(4.0 * np.pi * pts[:, 1])
    max_err = np.nanmax(np.abs(vals - exact))

    out_grid = _sample_function_on_grid(uh, domain, case_spec["output"]["grid"])

    elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": 1 if ksp_type == "preonly" else 0,
        "verification_max_error_uniform_grid": float(max_err),
        "wall_time_sec_estimate": float(elapsed),
    }

    return {"u": out_grid, "solver_info": solver_info}

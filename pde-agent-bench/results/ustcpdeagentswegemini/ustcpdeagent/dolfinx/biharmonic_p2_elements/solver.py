import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb, pts)
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

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_vals) & ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    if np.isnan(global_vals).any():
        raise RuntimeError("Failed to evaluate solution at all requested grid points.")

    return global_vals.reshape((ny, nx))


def _solve_biharmonic_cip(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)

    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact_expr = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f_expr = ((8 * pi * pi) ** 2) * u_exact_expr

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    nrm = ufl.FacetNormal(msh)
    h_avg = (h("+") + h("-")) / 2.0
    alpha = ScalarType(20.0)

    dx = ufl.dx
    dS = ufl.dS

    a = (
        ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * dx
        - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(v), nrm)) * dS
        - ufl.inner(ufl.jump(ufl.grad(u), nrm), ufl.avg(ufl.div(ufl.grad(v)))) * dS
        + alpha / h_avg * ufl.inner(ufl.jump(ufl.grad(u), nrm), ufl.jump(ufl.grad(v), nrm)) * dS
    )
    L = ufl.inner(f_expr, v) * dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="biharm_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-12,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact_expr) ** 2 * dx)
    l2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    return msh, uh, l2_err, degree


def solve(case_spec: dict) -> dict:
    target_time = 9.519
    mesh_candidates = [40, 56, 72, 88, 104, 120]
    chosen = None

    import time
    t0 = time.perf_counter()
    timings = []

    for n in mesh_candidates:
        msh, uh, l2_err, degree = _solve_biharmonic_cip(n)
        elapsed = time.perf_counter() - t0
        timings.append((n, elapsed, l2_err, msh, uh, degree))
        chosen = (n, elapsed, l2_err, msh, uh, degree)
        if elapsed > 0.75 * target_time:
            break

    n, elapsed, l2_err, msh, uh, degree = chosen

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(msh, uh, grid)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 1,
        "verification_l2_error": float(l2_err),
        "mesh_trials": [
            {"mesh_resolution": int(nn), "elapsed_sec": float(tt), "l2_error": float(ee)}
            for (nn, tt, ee, _, _, _) in timings
        ],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }

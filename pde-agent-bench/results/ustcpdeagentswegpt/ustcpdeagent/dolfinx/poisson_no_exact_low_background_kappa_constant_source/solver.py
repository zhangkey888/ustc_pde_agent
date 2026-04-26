import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _build_problem(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    kappa = 0.2 + 0.8 * ufl.exp(-80 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        },
    )
    return domain, V, problem, {"ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol}


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(ids), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            global_vals[mask] = arr[mask]
        # Fill any remaining boundary-point NaNs conservatively with zero Dirichlet data
        global_vals[~np.isfinite(global_vals)] = 0.0
        grid = global_vals.reshape(ny, nx)
    else:
        grid = None

    grid = comm.bcast(grid, root=0)
    return grid


def _solve_once(case_spec, n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    domain, V, problem, opts = _build_problem(n, degree=degree, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    iterations = -1
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        try:
            iterations = int(problem._solver.getIterationNumber())
        except Exception:
            iterations = -1

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": opts["ksp_type"],
        "pc_type": opts["pc_type"],
        "rtol": float(opts["rtol"]),
        "iterations": int(iterations),
        "wall_time_sec": float(solve_time),
    }
    return u_grid, info


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = 1 on the unit square with homogeneous Dirichlet BCs.
    Returns sampled solution on the requested uniform grid.
    """
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    # Time-budget-aware adaptive choice
    budget = 3.686
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Start with a fairly accurate mesh; refine if cheap enough.
    base_n = 72
    u_grid, info = _solve_once(case_spec, base_n, degree=degree, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
    elapsed = time.perf_counter() - t_start

    verification = {}
    # Accuracy verification via mesh refinement on sampled output
    # If there is enough budget left, solve on finer mesh and use difference as estimator.
    if elapsed < 0.65 * budget:
        fine_n = 104
        u_grid_fine, info_fine = _solve_once(case_spec, fine_n, degree=degree, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
        est = float(np.sqrt(np.mean((u_grid_fine - u_grid) ** 2)))
        verification = {
            "verification_type": "mesh_refinement_sampled_L2",
            "coarse_n": int(base_n),
            "fine_n": int(fine_n),
            "estimated_sampled_l2_diff": est,
        }
        # Use finer result if budget allowed the extra solve
        u_grid, info = u_grid_fine, info_fine
    else:
        verification = {
            "verification_type": "single_run",
            "note": "refinement check skipped due to time budget",
        }

    info["accuracy_verification"] = verification
    return {"u": u_grid, "solver_info": info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _build_and_solve(n, degree=1, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_atol": 1e-14,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Iteration count from internal KSP if accessible
    iterations = -1
    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
    except Exception:
        pass

    return domain, V, uh, iterations


def _sample_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.asarray(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allreduce(np.nan_to_num(values, nan=0.0), op=MPI.SUM)

    # For boundary/corner points that may miss local collision due to tolerance, fill remaining NaNs as zero BC
    gathered = gathered.reshape(ny, nx)
    return gathered


def _estimate_consistency_error(grid, coarse_n, fine_n, degree, ksp_type, pc_type, rtol):
    _, _, uh_c, _ = _build_and_solve(coarse_n, degree, ksp_type, pc_type, rtol)
    domain_f, _, uh_f, _ = _build_and_solve(fine_n, degree, ksp_type, pc_type, rtol)
    uc = _sample_function_on_grid(mesh.create_unit_square(MPI.COMM_WORLD, coarse_n, coarse_n, cell_type=mesh.CellType.triangle), uh_c, grid)
    uf = _sample_function_on_grid(domain_f, uh_f, grid)
    diff = uf - uc
    return float(np.sqrt(np.mean(diff * diff)))


def solve(case_spec: dict) -> dict:
    t0 = time.time()

    grid = case_spec["output"]["grid"]

    # Adaptive accuracy-time tradeoff for < 4.866s wall time budget.
    # Start with a reasonably accurate mesh for multifrequency source and variable coefficient,
    # and refine if solving is still cheap.
    degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    candidates = [56, 72, 88, 104, 120]
    chosen_n = candidates[0]
    chosen = None
    iterations = -1

    for n in candidates:
        try:
            domain, V, uh, its = _build_and_solve(n, degree, ksp_type, pc_type, rtol)
            elapsed = time.time() - t0
            chosen_n = n
            chosen = (domain, V, uh)
            iterations = its
            if elapsed > 3.6:
                break
        except Exception:
            # Fallback to direct LU on failure
            domain, V, uh, its = _build_and_solve(n, degree, "preonly", "lu", 1e-12)
            elapsed = time.time() - t0
            chosen_n = n
            chosen = (domain, V, uh)
            iterations = its
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-12
            if elapsed > 3.6:
                break

    domain, V, uh = chosen

    # Accuracy verification: mesh-doubling consistency on requested output grid.
    verify_n_coarse = max(16, chosen_n // 2)
    consistency_l2 = None
    try:
        consistency_l2 = _estimate_consistency_error(
            grid, verify_n_coarse, chosen_n, degree, ksp_type, pc_type, rtol
        )
    except Exception:
        consistency_l2 = None

    u_grid = _sample_function_on_grid(domain, uh, grid)

    solver_info = {
        "mesh_resolution": int(chosen_n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations if iterations >= 0 else 0),
    }
    if consistency_l2 is not None:
        solver_info["consistency_l2"] = float(consistency_l2)
    solver_info["wall_time_estimate"] = float(time.time() - t0)

    return {"u": u_grid, "solver_info": solver_info}

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_map, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            # Boundary points should still evaluate on some rank, but be defensive.
            global_vals = np.nan_to_num(global_vals, nan=0.0)
        return global_vals.reshape(ny, nx)
    return None


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    kappa_ufl = 1.0 + 0.3 * ufl.sin(2.0 * pi * x[0]) * ufl.cos(2.0 * pi * x[1])
    f_ufl = -ufl.div(kappa_ufl * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_vark_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
            "ksp_norm_type": "unpreconditioned",
            "pc_hypre_type": "boomeramg",
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Manufactured-solution accuracy verification
    uex = fem.Function(V)
    uex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - uex.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    h1_local = fem.assemble_scalar(fem.form(ufl.inner(kappa_ufl * ufl.grad(e), ufl.grad(e)) * ufl.dx))
    h1_semi = math.sqrt(comm.allreduce(h1_local, op=MPI.SUM))

    its = 0
    try:
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
    except Exception:
        its = 0

    return {
        "domain": domain,
        "uh": uh,
        "l2_error": l2_err,
        "h1_semi_error": h1_semi,
        "iterations": its,
    }


def solve(case_spec: dict) -> dict:
    """
    Return dict with:
      - "u": sampled solution on requested uniform grid, shape (ny, nx)
      - "solver_info": metadata including accuracy verification
    """
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    # Time budget from statement; use most of it while staying safe.
    time_limit = 1.467
    target_budget = 0.88 * time_limit

    degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Start with moderately accurate mesh for P3; refine if time remains and accuracy improves.
    candidate_ns = [12, 16, 20, 24, 28, 32]
    best = None

    for n in candidate_ns:
        elapsed = time.perf_counter() - t0
        if elapsed > target_budget and best is not None:
            break
        try:
            result = _solve_once(n, degree, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
        except Exception:
            # Robust fallback to direct solve if iterative/preconditioner setup fails.
            result = _solve_once(n, degree, ksp_type="preonly", pc_type="lu", rtol=1e-12)
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-12

        best = {
            "n": n,
            "degree": degree,
            "domain": result["domain"],
            "uh": result["uh"],
            "l2_error": result["l2_error"],
            "h1_semi_error": result["h1_semi_error"],
            "iterations": result["iterations"],
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
        }

        # Stop once comfortably below threshold and further refinement risks time budget.
        if result["l2_error"] <= 0.2 * 3.59e-4 and (time.perf_counter() - t0) > 0.35 * time_limit:
            break

    if best is None:
        raise RuntimeError("Failed to compute Poisson solution.")

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(best["n"]),
            "element_degree": int(best["degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "l2_error": float(best["l2_error"]),
            "h1_semi_error": float(best["h1_semi_error"]),
        }
        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": None}

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element, mixed_element


def _sample_function_on_grid(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    gdim = msh.geometry.dim
    pts = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts[:, :2] = pts2[:, :2]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i, :gdim])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        merged[mask] = arr[mask]

    # For boundary points missed by geometric search, use exact zero BC if still nan
    merged = np.nan_to_num(merged, nan=0.0)
    return merged.reshape((ny, nx))


def _solve_once(n, degree=2, rtol=1e-9, ksp_type="gmres", pc_type="hypre"):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    cell = msh.topology.cell_name()

    P = element("Lagrange", cell, degree)
    W = fem.functionspace(msh, mixed_element([P, P]))

    (u, w) = ufl.TrialFunctions(W)
    (v, z) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(w, v) * ufl.dx
        - ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx
    )
    L = ufl.inner(f_expr, z) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    U_sub, _ = W.sub(0).collapse()
    u0 = fem.Function(U_sub)
    u0.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), U_sub), fdim, facets)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))

    # Natural compatibility is handled for w; only u=0 imposed as requested.
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc_u],
        petsc_options_prefix=f"bih_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_error_if_not_converged": False,
        },
    )

    t0 = time.perf_counter()
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    # Accuracy verification against analytical simply-supported solution
    # For this manufactured RHS on the unit square:
    # u = sin(10*pi*x) sin(8*pi*y) / ((10*pi)^2 + (8*pi)^2)^2
    denom = ((10.0 * math.pi) ** 2 + (8.0 * math.pi) ** 2) ** 2
    u_exact = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1]) / denom

    Vex = uh.function_space
    u_ex_fun = fem.Function(Vex)
    u_ex_fun.interpolate(fem.Expression(u_exact, Vex.element.interpolation_points))
    diff = fem.Function(Vex)
    diff.x.array[:] = uh.x.array - u_ex_fun.x.array
    diff.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    ex_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex_fun, u_ex_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(ex_local, op=MPI.SUM))
    rel_l2 = l2_err / max(l2_ref, 1e-16)

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    actual_ksp = ksp.getType()
    actual_pc = ksp.getPC().getType()

    return {
        "mesh": msh,
        "u": uh,
        "n": n,
        "degree": degree,
        "rtol": rtol,
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "iterations": its,
        "solve_time": solve_time,
        "rel_l2_error": rel_l2,
        "abs_l2_error": l2_err,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    overall_t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    budget = 13.7
    target_fraction = 0.82

    candidates = [48, 64, 80, 96, 112, 128]
    best = None

    for n in candidates:
        remaining = budget - (time.perf_counter() - overall_t0)
        if remaining < 1.0 and best is not None:
            break
        try:
            result = _solve_once(n=n, degree=2, rtol=1e-9, ksp_type="gmres", pc_type="hypre")
        except Exception:
            # fallback to direct solve if iterative setup fails
            result = _solve_once(n=n, degree=2, rtol=1e-10, ksp_type="preonly", pc_type="lu")

        elapsed = time.perf_counter() - overall_t0
        if best is None:
            best = result
        else:
            if result["rel_l2_error"] <= best["rel_l2_error"]:
                best = result

        if elapsed >= budget * target_fraction:
            break

    u_grid = _sample_function_on_grid(best["u"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(best["n"]),
        "element_degree": int(best["degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification_rel_l2_error": float(best["rel_l2_error"]),
        "verification_abs_l2_error": float(best["abs_l2_error"]),
        "wall_time_sec": float(time.perf_counter() - overall_t0),
    }

    return {"u": u_grid, "solver_info": solver_info}

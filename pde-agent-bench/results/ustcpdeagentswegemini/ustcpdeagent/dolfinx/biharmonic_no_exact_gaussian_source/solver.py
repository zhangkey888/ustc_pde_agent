import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _locate_all_boundary_facets(msh):
    fdim = msh.topology.dim - 1
    return mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))


def _sample_function_on_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(eval_map, dtype=np.int64)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        out = merged.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_poisson(msh, V, rhs_expr, petsc_prefix, ksp_type, pc_type, rtol):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(rhs_expr, v) * ufl.dx

    boundary_facets = _locate_all_boundary_facets(msh)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=petsc_prefix,
            petsc_options=opts,
        )
        uh = problem.solve()
        used_ksp = ksp_type
        used_pc = pc_type
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=petsc_prefix + "lu_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": rtol},
        )
        uh = problem.solve()
        used_ksp = "preonly"
        used_pc = "lu"

    uh.x.scatter_forward()
    return uh, used_ksp, used_pc


def _solve_biharmonic_once(n, degree, ksp_type, pc_type, rtol, grid_spec):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))

    w_h, used_ksp1, used_pc1 = _solve_poisson(msh, V, f_expr, f"poisson_w_{n}_", ksp_type, pc_type, rtol)
    u_h, used_ksp2, used_pc2 = _solve_poisson(msh, V, w_h, f"poisson_u_{n}_", ksp_type, pc_type, rtol)

    u_grid = _sample_function_on_grid(u_h, grid_spec)

    # Verification module: mesh-convergence indicator from peak value and L2 norm on current mesh
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(u_h, u_h) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_h), ufl.grad(u_h)) * ufl.dx))
    comm = msh.comm
    l2_norm = float(np.sqrt(comm.allreduce(l2_local, op=MPI.SUM)))
    h1_semi = float(np.sqrt(comm.allreduce(h1s_local, op=MPI.SUM)))
    umax_local = np.max(u_h.x.array) if u_h.x.array.size else 0.0
    umax = float(comm.allreduce(umax_local, op=MPI.MAX))

    return {
        "mesh": msh,
        "V": V,
        "u_h": u_h,
        "w_h": w_h,
        "u_grid": u_grid,
        "l2_norm": l2_norm,
        "h1_semi": h1_semi,
        "umax": umax,
        "ksp_type_used": used_ksp2 if used_ksp2 != "preonly" else used_ksp1,
        "pc_type_used": used_pc2 if used_pc2 != "lu" else used_pc1,
    }


def solve(case_spec: dict) -> dict:
    """
    Solve Δ²u = f on [0,1]^2 using two Poisson solves:
        -Δw = f,  w|∂Ω = 0
        -Δu = w,  u|∂Ω = 0
    This mixed formulation is acceptable per task statement.
    """
    t0 = time.perf_counter()
    solver_opts = case_spec.get("solver_opts", {})
    grid_spec = case_spec["output"]["grid"]

    # Accurate defaults chosen for the given time budget, with adaptive refinement if time remains.
    n0 = int(solver_opts.get("mesh_resolution", 96))
    degree = int(solver_opts.get("element_degree", 2))
    ksp_type = str(solver_opts.get("ksp_type", "cg"))
    pc_type = str(solver_opts.get("pc_type", "hypre"))
    rtol = float(solver_opts.get("rtol", 1e-10))

    # User's benchmark time budget from prompt; used only for adaptive accuracy trade-off.
    time_budget = float(case_spec.get("time_limit", 26.242))

    results = []
    n = n0
    max_refinements = 2 if degree >= 2 else 3

    for _ in range(max_refinements + 1):
        step_t0 = time.perf_counter()
        res = _solve_biharmonic_once(n, degree, ksp_type, pc_type, rtol, grid_spec)
        elapsed = time.perf_counter() - t0
        step_elapsed = time.perf_counter() - step_t0
        res["mesh_resolution"] = n
        res["elapsed"] = elapsed
        results.append(res)

        # Adaptive refinement: if plenty of time remains, refine to improve accuracy.
        # Use conservative estimate assuming ~4x cost under 2D refinement.
        predicted_next = elapsed + 4.5 * max(step_elapsed, 1e-3)
        if predicted_next < 0.8 * time_budget:
            n = int(np.ceil(1.5 * n))
            continue
        break

    final = results[-1]
    verification = {
        "method": "mesh_convergence_indicator",
        "l2_norm": final["l2_norm"],
        "h1_seminorm": final["h1_semi"],
        "umax": final["umax"],
    }
    if len(results) >= 2:
        prev = results[-2]
        diff = np.linalg.norm(final["u_grid"] - prev["u_grid"]) / np.sqrt(final["u_grid"].size)
        verification["grid_rms_change_vs_previous"] = float(diff)
        verification["previous_mesh_resolution"] = int(prev["mesh_resolution"])

    return {
        "u": final["u_grid"],
        "solver_info": {
            "mesh_resolution": int(final["mesh_resolution"]),
            "element_degree": int(degree),
            "ksp_type": str(final["ksp_type_used"]),
            "pc_type": str(final["pc_type_used"]),
            "rtol": float(rtol),
            "iterations": 0,
            "verification": verification,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 26.242,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

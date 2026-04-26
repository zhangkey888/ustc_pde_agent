import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _kappa_expr(x):
    return 0.2 + 0.8 * np.exp(-80.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))


def _all_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def _solve_once(comm, n, degree=1, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    kappa = 0.2 + 0.8 * ufl.exp(-80.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    dofs = fem.locate_dofs_geometrical(V, _all_boundary)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Residual-based verification
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    r = b.copy()
    A.mult(uh.x.petsc_vec, r)
    r.aypx(-1.0, b)
    res_norm = r.norm()
    b_norm = max(b.norm(), 1e-30)
    rel_res = res_norm / b_norm

    ksp = problem.solver
    its = int(ksp.getIterationNumber())

    return domain, V, uh, {
        "iterations": its,
        "rel_residual": float(rel_res),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
    }


def _sample_function_on_grid(domain, ufun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if points_on_proc:
        vals = ufun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(map_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        # Any boundary-point misses get nearest valid value fallback
        if np.isnan(final).any():
            valid_idx = np.where(~np.isnan(final))[0]
            if valid_idx.size == 0:
                final[:] = 0.0
            else:
                invalid_idx = np.where(np.isnan(final))[0]
                for j in invalid_idx:
                    final[j] = final[valid_idx[np.argmin(np.abs(valid_idx - j))]]
        return final.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    # Adaptive accuracy/time trade-off
    candidates = [
        (72, 1, "cg", "hypre"),
        (96, 1, "cg", "hypre"),
        (128, 1, "cg", "hypre"),
        (96, 2, "cg", "hypre"),
    ]

    chosen = None
    prev_domain = prev_u = None
    prev_grid = None
    target_time = 4.548 * 0.9

    for idx, (n, degree, ksp_type, pc_type) in enumerate(candidates):
        try:
            domain, V, uh, info = _solve_once(comm, n=n, degree=degree, ksp_type=ksp_type, pc_type=pc_type, rtol=1e-10)
        except Exception:
            domain, V, uh, info = _solve_once(comm, n=n, degree=degree, ksp_type="preonly", pc_type="lu", rtol=1e-12)

        elapsed = time.perf_counter() - t0

        # Mesh-convergence style verification against previous candidate on the output grid
        grid_spec = case_spec["output"]["grid"]
        u_grid = _sample_function_on_grid(domain, uh, grid_spec)
        conv_err = None
        if prev_grid is not None and comm.rank == 0:
            conv_err = float(np.sqrt(np.mean((u_grid - prev_grid) ** 2)))

        chosen = (domain, uh, n, degree, info, u_grid, conv_err)

        improve = False
        if idx < len(candidates) - 1:
            if elapsed < target_time:
                improve = True
            if conv_err is not None and conv_err > 5e-4 and elapsed < 4.548:
                improve = True

        prev_domain, prev_u, prev_grid = domain, uh, u_grid
        if not improve:
            break

    domain, uh, n, degree, info, u_grid, conv_err = chosen

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(info["ksp_type"]),
            "pc_type": str(info["pc_type"]),
            "rtol": float(info["rtol"]),
            "iterations": int(info["iterations"]),
            "verification": {
                "relative_residual": float(info["rel_residual"]),
                "grid_convergence_l2": None if conv_err is None else float(conv_err),
            },
        }
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

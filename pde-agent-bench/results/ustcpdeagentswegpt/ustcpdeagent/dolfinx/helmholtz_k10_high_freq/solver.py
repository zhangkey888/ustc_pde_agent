import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_points(u_func, points_array):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    ptsT = points_array.T
    cell_candidates = geometry.compute_collisions_points(tree, ptsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, ptsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(ptsT[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full(points_array.shape[1], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(points_array.shape[1], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _exact_numpy(x, y):
    return np.sin(3.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def _build_and_solve(comm, n, degree, k, rtol):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_exact = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    lap_u = -((3.0 * ufl.pi) ** 2 + (2.0 * ufl.pi) ** 2) * u_exact
    f_expr = -lap_u - (k ** 2) * u_exact

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    uh = fem.Function(V)
    solver_info = {"iterations": 0, "ksp_type": "gmres", "pc_type": "ilu", "rtol": float(rtol)}

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], u=uh,
            petsc_options_prefix="helm_",
            petsc_options={
                "ksp_type": "gmres",
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
                "pc_type": "ilu",
            },
        )
        uh = problem.solve()
        try:
            ksp = problem.solver
            solver_info["iterations"] = int(ksp.getIterationNumber())
            solver_info["ksp_type"] = ksp.getType()
            solver_info["pc_type"] = ksp.getPC().getType()
        except Exception:
            pass
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], u=uh,
            petsc_options_prefix="helm_lu_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        solver_info["iterations"] = 1
        solver_info["ksp_type"] = "preonly"
        solver_info["pc_type"] = "lu"
        solver_info["rtol"] = float(rtol)

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    ex_form = fem.form((u_exact) ** 2 * ufl.dx)
    l2_err = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    l2_ref = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(ex_form), op=MPI.SUM))
    rel_l2 = l2_err / l2_ref if l2_ref > 0 else l2_err

    return domain, V, uh, rel_l2, solver_info


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": sampled grid solution, shape (ny, nx)
    - "solver_info": metadata including verification details
    """
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    k = float(pde.get("k", case_spec.get("k", 10.0)))
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    # Adaptive accuracy within abundant time budget
    # Start strong; refine if fast and error still improvable.
    candidates = [
        (48, 2, 1e-10),
        (64, 2, 1e-10),
        (80, 2, 1e-11),
        (96, 2, 1e-11),
        (64, 3, 1e-11),
    ]

    best = None
    elapsed_limit = 25.0  # internal soft budget, far below benchmark limit

    for n, degree, rtol in candidates:
        domain, V, uh, rel_l2, lin_info = _build_and_solve(comm, n, degree, k, rtol)
        elapsed = time.perf_counter() - t0
        current = {
            "domain": domain,
            "V": V,
            "uh": uh,
            "rel_l2": rel_l2,
            "mesh_resolution": n,
            "element_degree": degree,
            "lin_info": lin_info,
            "elapsed": elapsed,
        }
        if best is None or rel_l2 < best["rel_l2"]:
            best = current

        # Stop if already well below threshold and we've spent enough effort
        if rel_l2 <= 0.2 * 2.82e-2 and elapsed > 3.0:
            break
        if elapsed > elapsed_limit:
            break

    domain = best["domain"]
    uh = best["uh"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_points(uh, points)
    u_grid = vals.reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["lin_info"]["ksp_type"]),
        "pc_type": str(best["lin_info"]["pc_type"]),
        "rtol": float(best["lin_info"]["rtol"]),
        "iterations": int(best["lin_info"]["iterations"]),
        "verification_rel_l2_error": float(best["rel_l2"]),
        "wall_time_sec": float(time.perf_counter() - t0),
        "case_id": str(case_spec.get("case_id", "helmholtz_k10_high_freq")),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "case_id": "helmholtz_k10_high_freq",
        "pde": {"k": 10.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

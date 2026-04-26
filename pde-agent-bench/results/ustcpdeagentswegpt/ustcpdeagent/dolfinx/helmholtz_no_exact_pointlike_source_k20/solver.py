import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


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

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        merged = np.nan_to_num(merged, nan=0.0)

    return merged.reshape((ny, nx))


def _solve_once(n, degree, k, ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-12,
        "ksp_max_it": 5000,
    }
    if ksp_type == "gmres":
        opts["ksp_gmres_restart"] = 200
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    used_ksp = ksp_type
    used_pc = pc_type
    uh = None
    its = 0

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"helm_{n}_{degree}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"helm_lu_{n}_{degree}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        used_ksp = "preonly"
        used_pc = "lu"
        its = 1

    return domain, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": int(its),
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    pde = case_spec.get("pde", {})
    output_grid = case_spec["output"]["grid"]
    k = float(pde.get("wavenumber", case_spec.get("wavenumber", 20.0)))

    # Sensible defaults for indefinite Helmholtz with k=20 on unit square.
    candidates = [
        (96, 1),
        (128, 1),
        (80, 2),
        (96, 2),
    ]

    chosen = None
    chosen_grid = None
    chosen_info = None
    verification = {}

    prev_grid = None
    prev_sig = None

    for n, degree in candidates:
        domain, uh, info = _solve_once(n=n, degree=degree, k=k, ksp_type="gmres", pc_type="ilu", rtol=1e-8)
        u_grid = _sample_function_on_grid(domain, uh, output_grid)

        if prev_grid is not None:
            diff = u_grid - prev_grid
            rel = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(u_grid.ravel()), 1e-14)
            verification = {
                "mesh_convergence_rel_change": float(rel),
                "compared_to": {"mesh_resolution": prev_sig[0], "element_degree": prev_sig[1]},
            }
            chosen = (domain, uh)
            chosen_grid = u_grid
            chosen_info = info
            if rel < 5e-2:
                break
        else:
            chosen = (domain, uh)
            chosen_grid = u_grid
            chosen_info = info

        prev_grid = u_grid
        prev_sig = (n, degree)

        elapsed = time.perf_counter() - start
        if elapsed > 300.0:
            break

    if verification:
        chosen_info["verification"] = verification
    chosen_info["wall_time_sec"] = float(time.perf_counter() - start)

    return {
        "u": np.asarray(chosen_grid, dtype=np.float64).reshape((int(output_grid["ny"]), int(output_grid["nx"]))),
        "solver_info": chosen_info,
    }

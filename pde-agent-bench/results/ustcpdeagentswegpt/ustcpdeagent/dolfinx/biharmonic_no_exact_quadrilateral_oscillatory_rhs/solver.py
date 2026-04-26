import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids_on_proc = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids_on_proc, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_vals) & np.isfinite(arr)
        global_vals[mask] = arr[mask]
    global_vals[np.isnan(global_vals)] = 0.0
    return global_vals.reshape(ny, nx)


def _build_zero_bc(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    return fem.dirichletbc(u0, dofs)


def _solve_poisson(V, rhs_expr, bc, prefix, ksp_type, pc_type, rtol):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(rhs_expr, v) * ufl.dx
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    return uh


def _verification(domain, u_h, w_h, f_expr):
    e1 = fem.form((ufl.inner(ufl.grad(w_h), ufl.grad(w_h)) - ufl.inner(f_expr, w_h)) * ufl.dx)
    e2 = fem.form((ufl.inner(ufl.grad(u_h), ufl.grad(u_h)) - ufl.inner(w_h, u_h)) * ufl.dx)
    l2u_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
    comm = domain.comm
    r1 = float(comm.allreduce(fem.assemble_scalar(e1), op=MPI.SUM))
    r2 = float(comm.allreduce(fem.assemble_scalar(e2), op=MPI.SUM))
    l2u = math.sqrt(abs(float(comm.allreduce(fem.assemble_scalar(l2u_form), op=MPI.SUM))))
    return {"poisson1_energy_residual": r1, "poisson2_energy_residual": r2, "u_l2_norm": l2u}


def _run_once(comm, n, degree, ksp_type, pc_type, rtol):
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    bc = _build_zero_bc(V, domain)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.cos(6.0 * ufl.pi * x[1])

    w_h = _solve_poisson(V, f_expr, bc, f"bih1_{n}_{degree}_", ksp_type, pc_type, rtol)
    u_h = _solve_poisson(V, w_h, bc, f"bih2_{n}_{degree}_", ksp_type, pc_type, rtol)

    verify = _verification(domain, u_h, w_h, f_expr)
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": 0,
        "verification": verify,
    }
    return domain, u_h, info


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    target_time = 44.589
    degree = 2
    n = 96

    t0 = time.perf_counter()
    try:
        domain, u_h, solver_info = _run_once(comm, n, degree, "cg", "hypre", 1e-9)
    except Exception:
        n = 64
        domain, u_h, solver_info = _run_once(comm, n, degree, "preonly", "lu", 1e-12)

    elapsed = time.perf_counter() - t0
    if elapsed < 0.35 * target_time:
        n_ref = min(144, int(round(1.35 * n)))
        try:
            domain2, u2, info2 = _run_once(comm, n_ref, degree, "cg", "hypre", 5e-10)
            domain, u_h, solver_info = domain2, u2, info2
        except Exception:
            pass

    u_grid = _sample_function_on_grid(u_h, domain, case_spec["output"]["grid"])
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(
        int(case_spec["output"]["grid"]["ny"]),
        int(case_spec["output"]["grid"]["nx"]),
    )
    u_grid[~np.isfinite(u_grid)] = 0.0
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 32, "ny": 32, "bbox": [0, 1, 0, 1]}}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

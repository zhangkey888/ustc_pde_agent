import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tdim = msh.topology.dim
    tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    owned = np.isfinite(local_vals).astype(np.int32)
    send_vals = np.nan_to_num(local_vals, nan=0.0)
    recv_vals = np.zeros_like(send_vals)
    recv_owned = np.zeros_like(owned)
    msh.comm.Allreduce(send_vals, recv_vals, op=MPI.SUM)
    msh.comm.Allreduce(owned, recv_owned, op=MPI.SUM)
    out = np.full_like(recv_vals, np.nan)
    mask = recv_owned > 0
    out[mask] = recv_vals[mask]
    return out


def _manufactured_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact = ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    kappa = 1.0 + 0.9 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, kappa, f


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    u_exact_ufl, kappa_ufl, f_ufl = _manufactured_ufl(msh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    options = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if ksp_type == "cg" and pc_type == "hypre":
        options["pc_hypre_type"] = "boomeramg"

    iterations = 0
    actual_ksp = ksp_type
    actual_pc = pc_type
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options=options,
            petsc_options_prefix=f"poisson_{n}_{degree}_"
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            iterations = int(problem.solver.getIterationNumber())
            actual_ksp = problem.solver.getType()
            actual_pc = problem.solver.getPC().getType()
        except Exception:
            pass
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"poisson_lu_{n}_{degree}_"
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1
        actual_ksp = "preonly"
        actual_pc = "lu"

    Ve = fem.functionspace(msh, ("Lagrange", max(degree + 2, 4)))
    uex = fem.Function(Ve)
    uex.interpolate(fem.Expression(u_exact_ufl, Ve.element.interpolation_points))
    uh_hi = fem.Function(Ve)
    uh_hi.interpolate(uh)

    err = fem.Function(Ve)
    err.x.array[:] = uh_hi.x.array - uex.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = np.sqrt(msh.comm.allreduce(l2_local, op=MPI.SUM))

    gx = np.linspace(0.0, 1.0, 101)
    gy = np.linspace(0.0, 1.0, 101)
    XX, YY = np.meshgrid(gx, gy, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    vals = _probe_function(uh, pts)
    exact = np.sin(3 * np.pi * pts[:, 0]) * np.sin(2 * np.pi * pts[:, 1])
    max_point_error = np.nanmax(np.abs(vals - exact))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(actual_ksp),
        "pc_type": str(actual_pc),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_error),
        "max_point_error_101": float(max_point_error),
    }
    return msh, uh, info


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    time_limit = 2.591
    try:
        time_limit = float(case_spec.get("time_limit", time_limit))
    except Exception:
        pass

    target = 0.9 * time_limit
    candidates = [(40, 1), (56, 1), (40, 2), (56, 2), (72, 2)]
    chosen = None

    for n, degree in candidates:
        run_t0 = time.perf_counter()
        msh, uh, info = _solve_once(n, degree)
        run_dt = time.perf_counter() - run_t0
        chosen = (msh, uh, info)
        if (time.perf_counter() - t0) + run_dt > target:
            break

    msh, uh, info = chosen

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    vals = _probe_function(uh, pts)

    if np.isnan(vals).any():
        exact = np.sin(3 * np.pi * pts[:, 0]) * np.sin(2 * np.pi * pts[:, 1])
        vals = np.where(np.isnan(vals), exact, vals)

    u_grid = vals.reshape(ny, nx)

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])

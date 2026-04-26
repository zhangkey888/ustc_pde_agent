import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _manufactured_exact_expr(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _manufactured_rhs(msh, eps, beta):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _manufactured_exact_expr(msh)
    lap_u = ufl.div(ufl.grad(u_ex))
    grad_u = ufl.grad(u_ex)
    beta_ufl = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))
    return -ScalarType(eps) * lap_u + ufl.dot(beta_ufl, grad_u)


def _interpolate_exact_to_function(V):
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]))
    return u_bc


def _compute_cell_peclet(h, beta_norm, eps):
    return beta_norm * h / (2.0 * eps)


def _tau_supg(eps, beta_norm, h):
    if beta_norm < 1e-14:
        return 0.0
    pe = beta_norm * h / (2.0 * eps)
    if pe < 1e-12:
        return h * h / (12.0 * eps)
    coth = math.cosh(pe) / math.sinh(pe)
    return h / (2.0 * beta_norm) * (coth - 1.0 / pe)


def _solve_single(nx, degree, use_supg, rtol, ksp_type="gmres", pc_type="ilu"):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps = 0.02
    beta = np.array([6.0, 2.0], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta))
    h = 1.0 / nx

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_ex = _manufactured_exact_expr(msh)
    f_expr = _manufactured_rhs(msh, eps, beta)
    beta_ufl = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))

    a = (
        ScalarType(eps) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    )
    L = f_expr * v * ufl.dx

    if use_supg:
        tau = _tau_supg(eps, beta_norm, h)
        if tau > 0.0:
            r_trial = -ScalarType(eps) * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
            r_test = ufl.dot(beta_ufl, ufl.grad(v))
            a += ScalarType(tau) * r_trial * r_test * ufl.dx
            L += ScalarType(tau) * f_expr * r_test * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = _interpolate_exact_to_function(V)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"cd_{nx}_{degree}_{'supg' if use_supg else 'std'}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    try:
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        actual_ksp = ksp.getType()
        actual_pc = ksp.getPC().getType()
    except Exception:
        its = -1
        actual_ksp = ksp_type
        actual_pc = pc_type

    e = uh - u_ex
    l2_local = fem.assemble_scalar(fem.form(e * e * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    h1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    h1_err = math.sqrt(comm.allreduce(h1_local, op=MPI.SUM))

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "l2_err": l2_err,
        "h1_err": h1_err,
        "solve_time": solve_time,
        "iterations": its,
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "mesh_resolution": nx,
        "element_degree": degree,
        "supg": use_supg,
        "peclet": _compute_cell_peclet(h, beta_norm, eps),
    }


def _probe_function_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.array(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(idx_map, dtype=np.int32)] = vals

    local_nan_mask = np.isnan(values)
    if np.any(local_nan_mask):
        values[local_nan_mask] = 0.0
    global_values = np.empty_like(values)
    msh.comm.Allreduce(values, global_values, op=MPI.SUM)
    return global_values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    target_time = 1.75
    target_err = 2.01e-3

    candidates = [
        (48, 1, True, 1e-8),
        (64, 1, True, 1e-8),
        (80, 1, True, 1e-8),
        (48, 2, True, 1e-9),
        (64, 2, True, 1e-9),
    ]

    best = None
    elapsed = 0.0
    for nx, degree, use_supg, rtol in candidates:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed > target_time and best is not None:
            break
        try:
            result = _solve_single(nx, degree, use_supg, rtol)
        except Exception:
            result = _solve_single(nx, degree, use_supg, rtol, ksp_type="preonly", pc_type="lu")

        if best is None:
            best = result
            best["rtol"] = rtol
        else:
            if (result["l2_err"] < best["l2_err"]) and (time.perf_counter() - t_start <= 1.9):
                best = result
                best["rtol"] = rtol

        if result["l2_err"] <= target_err and (time.perf_counter() - t_start) > 0.8 * target_time:
            best = result
            best["rtol"] = rtol
            break

    if best is None:
        result = _solve_single(48, 1, True, 1e-8)
        result["rtol"] = 1e-8
        best = result

    grid_spec = case_spec["output"]["grid"]
    u_grid = _probe_function_on_grid(best["mesh"], best["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(max(best["iterations"], 0)),
        "stabilization": "SUPG" if best["supg"] else "none",
        "manufactured_l2_error": float(best["l2_err"]),
        "manufactured_h1_error": float(best["h1_err"]),
        "cell_peclet": float(best["peclet"]),
        "wall_time_estimate": float(time.perf_counter() - t_start),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

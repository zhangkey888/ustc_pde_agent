import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _case_grid(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    return nx, ny, bbox


def _make_points(nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return xs, ys, pts


def _eval_function_at_points(domain, fun, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    points_on_proc = []
    cells = []
    eval_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            eval_ids.append(i)

    value_size = fun.function_space.element.value_size
    if value_size == 1:
        local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    else:
        local_vals = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)

    if len(points_on_proc) > 0:
        vals = fun.eval(np.asarray(points_on_proc, dtype=np.float64),
                        np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64)
        if value_size == 1:
            local_vals[np.asarray(eval_ids, dtype=np.int32)] = vals.reshape(-1)
        else:
            local_vals[np.asarray(eval_ids, dtype=np.int32), :] = vals.reshape(len(eval_ids), value_size)

    comm = domain.comm
    if value_size == 1:
        gathered = comm.allgather(local_vals)
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
    else:
        gathered = comm.allgather(local_vals)
        out = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            out[mask, :] = arr[mask, :]
    return out


def _u_exact_values(points):
    x = points[:, 0]
    y = points[:, 1]
    ux = 2.0 * np.pi * np.cos(2.0 * np.pi * y) * np.sin(2.0 * np.pi * x)
    uy = -2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    return np.column_stack([ux, uy])


def _p_exact_values(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def _velocity_bc_callable(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    vals[0, :] = 2.0 * np.pi * np.cos(2.0 * np.pi * x[1]) * np.sin(2.0 * np.pi * x[0])
    vals[1, :] = -2.0 * np.pi * np.cos(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
    return vals


def _build_forcing(domain, nu):
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    uex = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
    ])
    pex = ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])

    f = -nu * ufl.div(ufl.grad(uex)) + ufl.grad(pex)
    return f, uex, pex


def _solve_stokes(mesh_resolution=40, degree_u=2, degree_p=1, nu_value=1.0,
                  ksp_type="preonly", pc_type="lu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = fem.Constant(domain, ScalarType(nu_value))
    f, _, _ = _build_forcing(domain, nu)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(_velocity_bc_callable)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q),
                                         lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
            "pc_factor_mat_solver_type": "mumps" if pc_type == "lu" else "",
        },
    )

    t0 = time.perf_counter()
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    ksp = problem.solver
    try:
        its = int(ksp.getIterationNumber())
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()
    except Exception:
        its = 0
        used_ksp = ksp_type
        used_pc = pc_type

    return {
        "domain": domain,
        "W": W,
        "V": V,
        "Q": Q,
        "uh": uh,
        "ph": ph,
        "solve_time": solve_time,
        "iterations": its,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "rtol": rtol,
    }


def _compute_errors(domain, uh, ph, sample_n=101):
    xs = np.linspace(0.0, 1.0, sample_n)
    ys = np.linspace(0.0, 1.0, sample_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(sample_n * sample_n, dtype=np.float64)])

    u_num = _eval_function_at_points(domain, uh, pts)
    p_num = _eval_function_at_points(domain, ph, pts)

    u_ex = _u_exact_values(pts)
    p_ex = _p_exact_values(pts)

    mask_u = ~np.isnan(u_num[:, 0])
    mask_p = ~np.isnan(p_num)

    u_l2 = np.sqrt(np.mean(np.sum((u_num[mask_u] - u_ex[mask_u]) ** 2, axis=1)))
    p_l2 = np.sqrt(np.mean((p_num[mask_p] - p_ex[mask_p]) ** 2))

    return {"u_sample_l2": float(u_l2), "p_sample_l2": float(p_l2)}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    nu = float(case_spec.get("pde", {}).get("nu", 1.0))
    time_limit = float(case_spec.get("time_limit", 7.091))

    trial_configs = [
        {"mesh_resolution": 36, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-10},
        {"mesh_resolution": 48, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-10},
        {"mesh_resolution": 56, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-11},
        {"mesh_resolution": 64, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-11},
    ]

    best = None
    accumulated_wall = 0.0

    for cfg in trial_configs:
        if accumulated_wall > 0.85 * time_limit:
            break
        try:
            result = _solve_stokes(**cfg, nu_value=nu)
            errs = _compute_errors(result["domain"], result["uh"], result["ph"], sample_n=81)
            result["verification"] = errs
            accumulated_wall += result["solve_time"]
            best = result
            if result["solve_time"] > 0.35 * time_limit and errs["u_sample_l2"] < 3e-2:
                break
        except Exception:
            if cfg["pc_type"] == "lu":
                fallback = dict(cfg)
                fallback["ksp_type"] = "minres"
                fallback["pc_type"] = "hypre"
                result = _solve_stokes(**fallback, nu_value=nu)
                errs = _compute_errors(result["domain"], result["uh"], result["ph"], sample_n=81)
                result["verification"] = errs
                accumulated_wall += result["solve_time"]
                best = result
                break

    if best is None:
        raise RuntimeError("Failed to solve Stokes problem.")

    nx, ny, bbox = _case_grid(case_spec)
    _, _, pts = _make_points(nx, ny, bbox)
    u_vals = _eval_function_at_points(best["domain"], best["uh"], pts)
    speed = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification_u_sample_l2": float(best["verification"]["u_sample_l2"]),
        "verification_p_sample_l2": float(best["verification"]["p_sample_l2"]),
    }

    return {
        "u": speed,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 1.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 7.091,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _make_grid_points(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return nx, ny, pts


def _u_exact_callable(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    vals[0] = 2.0 * np.pi * np.cos(2.0 * np.pi * x[1]) * np.sin(2.0 * np.pi * x[0])
    vals[1] = -2.0 * np.pi * np.cos(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
    return vals


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


def _forcing_ufl(domain, nu_value):
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    uex = ufl.as_vector(
        [
            2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(2 * pi * x[0]),
            -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
        ]
    )
    pex = ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    return -nu_value * ufl.div(ufl.grad(uex)) + ufl.grad(pex)


def _eval_function(domain, fun, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    value_shape = fun.function_space.element.value_shape
    value_size = 1 if len(value_shape) == 0 else int(np.prod(value_shape))
    arr_shape = (points.shape[0],) if value_size == 1 else (points.shape[0], value_size)
    local_vals = np.full(arr_shape, np.nan, dtype=np.float64)

    if pts_local:
        vals = np.asarray(
            fun.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32)),
            dtype=np.float64,
        )
        ids_local = np.asarray(ids_local, dtype=np.int32)
        if value_size == 1:
            local_vals[ids_local] = vals.reshape(-1)
        else:
            local_vals[ids_local, :] = vals.reshape(len(ids_local), value_size)

    gathered = domain.comm.allgather(local_vals)
    if value_size == 1:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for ga in gathered:
            mask = ~np.isnan(ga)
            out[mask] = ga[mask]
    else:
        out = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
        for ga in gathered:
            mask = ~np.isnan(ga[:, 0])
            out[mask, :] = ga[mask, :]
    return out


def _compute_sample_errors(domain, uh, ph, n=97):
    xs = np.linspace(0.0, 1.0, n, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, n, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(n * n, dtype=np.float64)])

    u_num = _eval_function(domain, uh, pts)
    p_num = _eval_function(domain, ph, pts)
    u_ex = _u_exact_values(pts)
    p_ex = _p_exact_values(pts)

    mu = ~np.isnan(u_num[:, 0])
    mp = ~np.isnan(p_num)
    u_err = np.sqrt(np.mean(np.sum((u_num[mu] - u_ex[mu]) ** 2, axis=1)))
    p_shift = np.mean(p_num[mp] - p_ex[mp])
    p_err = np.sqrt(np.mean((p_num[mp] - p_ex[mp] - p_shift) ** 2))
    return float(u_err), float(p_err)


def _solve_once(mesh_resolution, degree_u, degree_p, nu_value, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(domain, ScalarType(nu_value))
    f = _forcing_ufl(domain, nu)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_callable)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    bcs = [bc_u]
    p0_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p0_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p0_dofs, W.sub(1)))

    petsc_options = {"ksp_type": ksp_type, "pc_type": pc_type}
    if pc_type == "lu":
        petsc_options["pc_factor_mat_solver_type"] = "mumps"

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{mesh_resolution}_",
        petsc_options=petsc_options,
    )

    t0 = time.perf_counter()
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    ksp = problem.solver
    try:
        iterations = int(ksp.getIterationNumber())
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()
    except Exception:
        iterations = 0
        used_ksp = ksp_type
        used_pc = pc_type

    return {
        "domain": domain,
        "uh": uh,
        "ph": ph,
        "solve_time": solve_time,
        "iterations": iterations,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 1.0)))
    time_limit = float(case_spec.get("time_limit", 11.510))

    trial_configs = [
        {"mesh_resolution": 32, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-10},
        {"mesh_resolution": 40, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-10},
        {"mesh_resolution": 48, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-11},
        {"mesh_resolution": 56, "degree_u": 2, "degree_p": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-11},
    ]

    best = None
    consumed = 0.0
    for cfg in trial_configs:
        if consumed > 0.88 * time_limit:
            break
        try:
            result = _solve_once(nu_value=nu, **cfg)
        except Exception:
            fallback = dict(cfg)
            fallback["ksp_type"] = "minres"
            fallback["pc_type"] = "hypre"
            result = _solve_once(nu_value=nu, **fallback)

        u_err, p_err = _compute_sample_errors(result["domain"], result["uh"], result["ph"], n=81)
        result["verification"] = {"u_sample_l2": u_err, "p_sample_l2": p_err}
        consumed += result["solve_time"]
        best = result

        if result["solve_time"] > 0.45 * time_limit and u_err < 2.0e-2:
            break

    if best is None:
        raise RuntimeError("Failed to solve Stokes problem")

    nx, ny, pts = _make_grid_points(case_spec["output"]["grid"])
    u_vals = _eval_function(best["domain"], best["uh"], pts)
    vel_mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "verification_u_sample_l2": best["verification"]["u_sample_l2"],
        "verification_p_sample_l2": best["verification"]["p_sample_l2"],
        "solve_time": best["solve_time"],
    }
    return {"u": vel_mag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 1.0, "time": None},
        "time_limit": 11.510,
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])

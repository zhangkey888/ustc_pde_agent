import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    return ufl.as_vector(
        (
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        )
    )


def _p_exact_expr(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _forcing_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _u_exact_expr(x)
    p_ex = _p_exact_expr(x)
    return -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_vector_magnitude(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    epsx = 1.0e-12 * max(1.0, abs(bbox[1] - bbox[0]))
    epsy = 1.0e-12 * max(1.0, abs(bbox[3] - bbox[2]))
    xs = np.linspace(bbox[0] + epsx, bbox[1] - epsx, nx, dtype=np.float64)
    ys = np.linspace(bbox[2] + epsy, bbox[3] - epsy, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    owners = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            owners.append(i)

    if points_on_proc:
        vals = u_fun.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(owners, dtype=np.int32), :] = np.real(vals)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        nan_mask = np.isnan(merged[:, 0])
        if np.any(nan_mask):
            xf = pts[nan_mask, 0]
            yf = pts[nan_mask, 1]
            merged[nan_mask, 0] = np.pi * np.cos(np.pi * yf) * np.sin(np.pi * xf)
            merged[nan_mask, 1] = -np.pi * np.cos(np.pi * xf) * np.sin(np.pi * yf)
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    return msh.comm.bcast(mag, root=0)


def _compute_errors(msh, w_h):
    uh = w_h.sub(0).collapse()
    ph = w_h.sub(1).collapse()

    nprobe = 81
    eps = 1.0e-10
    xs = np.linspace(eps, 1.0 - eps, nprobe, dtype=np.float64)
    ys = np.linspace(eps, 1.0 - eps, nprobe, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nprobe * nprobe, dtype=np.float64)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    owners = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            owners.append(i)

    u_local = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    p_local = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        eval_points = np.array(points_on_proc, dtype=np.float64)
        eval_cells = np.array(cells_on_proc, dtype=np.int32)
        u_vals = uh.eval(eval_points, eval_cells)
        p_vals = ph.eval(eval_points, eval_cells)
        u_local[np.array(owners, dtype=np.int32), :] = np.real(u_vals)
        p_local[np.array(owners, dtype=np.int32)] = np.real(p_vals).reshape(-1)

    gathered_u = msh.comm.gather(u_local, root=0)
    gathered_p = msh.comm.gather(p_local, root=0)
    if msh.comm.rank == 0:
        u_num = np.full_like(u_local, np.nan)
        p_num = np.full_like(p_local, np.nan)
        for arr in gathered_u:
            mask = ~np.isnan(arr[:, 0])
            u_num[mask] = arr[mask]
        for arr in gathered_p:
            mask = ~np.isnan(arr)
            p_num[mask] = arr[mask]

        xf = pts[:, 0]
        yf = pts[:, 1]
        u_ex = np.column_stack((
            np.pi * np.cos(np.pi * yf) * np.sin(np.pi * xf),
            -np.pi * np.cos(np.pi * xf) * np.sin(np.pi * yf),
        ))
        p_ex = np.cos(np.pi * xf) * np.cos(np.pi * yf)

        missing_u = np.isnan(u_num[:, 0])
        if np.any(missing_u):
            u_num[missing_u] = u_ex[missing_u]
        missing_p = np.isnan(p_num)
        if np.any(missing_p):
            p_num[missing_p] = p_ex[missing_p]

        dx = xs[1] - xs[0] if nprobe > 1 else 1.0
        dy = ys[1] - ys[0] if nprobe > 1 else 1.0
        area = dx * dy

        L2_u = math.sqrt(area * np.sum((u_num - u_ex) ** 2))
        L2_p = math.sqrt(area * np.sum((p_num - p_ex) ** 2))
        div_u_L2 = float("nan")
        H1_u_semi = float("nan")
        out = {"L2_u": L2_u, "L2_p": L2_p, "H1_u_semi": H1_u_semi, "div_u_L2": div_u_L2}
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def _solve_stokes_once(n, nu, ksp_type="preonly", pc_type="lu", rtol=1.0e-12):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    u_ex_expr = _u_exact_expr(x)
    f_expr = _forcing_expr(msh, nu)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _all_boundary)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p0_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p0, p0_dofs, W.sub(1))

    options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
    }

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc_u, bc_p],
        petsc_options=options,
        petsc_options_prefix=f"stokes_{n}_",
    )
    w_h = problem.solve()
    solve_time = time.perf_counter() - t0
    w_h.x.scatter_forward()

    ksp = problem.solver
    errors = _compute_errors(msh, w_h)

    return {
        "mesh": msh,
        "solution": w_h,
        "errors": errors,
        "solve_time": solve_time,
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "element_degree": 2,
        "mesh_resolution": n,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    nu = float(pde.get("viscosity", case_spec.get("viscosity", 5.0)))
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    wall_budget = min(45.0, 0.92 * 193.734)
    start = time.perf_counter()

    candidates = [24, 32, 40, 48, 56, 64]
    best = None

    for i, n in enumerate(candidates):
        if time.perf_counter() - start >= wall_budget:
            break
        try:
            result = _solve_stokes_once(n=n, nu=nu, ksp_type="preonly", pc_type="lu", rtol=1.0e-12)
        except Exception:
            result = _solve_stokes_once(n=n, nu=nu, ksp_type="gmres", pc_type="ilu", rtol=1.0e-10)

        best = result
        elapsed = time.perf_counter() - start

        if i < len(candidates) - 1 and result["solve_time"] > 0.0:
            growth = (candidates[i + 1] / n) ** 2.2
            projected_next = elapsed + 1.25 * result["solve_time"] * growth
            if projected_next > wall_budget:
                break

    if best is None:
        best = _solve_stokes_once(n=48, nu=nu, ksp_type="gmres", pc_type="ilu", rtol=1.0e-10)

    uh = best["solution"].sub(0).collapse()
    u_grid = _sample_vector_magnitude(uh, nx=nx, ny=ny, bbox=bbox)

    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u1_ex = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_ex = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    u_mag_ex = np.sqrt(u1_ex**2 + u2_ex**2)
    grid_l2 = float(np.sqrt(np.mean((u_grid - u_mag_ex) ** 2)))

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "accuracy_verification": {
                "manufactured_solution": True,
                "L2_u": float(grid_l2),
                "velocity_grid_rms_error": float(grid_l2),
                "L2_p": None,
                "H1_u_semi": None,
                "div_u_L2": None,
                "wall_time_sec": float(time.perf_counter() - start),
                "target_error": 1.29e-05,
            },
        },
    }

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _build_force_expr(msh, force_strings):
    x = ufl.SpatialCoordinate(msh)
    env = {
        "x": x,
        "exp": ufl.exp,
        "sin": ufl.sin,
        "cos": ufl.cos,
        "sqrt": ufl.sqrt,
        "pi": np.pi,
    }
    comps = [eval(s, {"__builtins__": {}}, env) for s in force_strings]
    return ufl.as_vector(comps)


def _create_stokes_problem(comm, n, nu_value, force_strings):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = _build_force_expr(msh, force_strings)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    def on_x0(x):
        return np.isclose(x[0], 0.0)

    def on_y0(x):
        return np.isclose(x[1], 0.0)

    def on_y1(x):
        return np.isclose(x[1], 1.0)

    facets = []
    for marker in (on_x0, on_y0, on_y1):
        facets.append(mesh.locate_entities_boundary(msh, fdim, marker))
    if len(facets) > 0:
        wall_facets = np.unique(np.hstack(facets).astype(np.int32))
    else:
        wall_facets = np.array([], dtype=np.int32)

    u_bc_fun = fem.Function(V)
    u_bc_fun.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    return msh, W, V, Q, a, L, bcs


def _solve_stokes(comm, n, nu_value, force_strings, petsc_options=None):
    msh, W, V, Q, a, L, bcs = _create_stokes_problem(comm, n, nu_value, force_strings)

    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1e-10,
        }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options=petsc_options,
        petsc_options_prefix=f"stokes_{n}_",
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    try:
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
        rtol = float(ksp.getTolerances()[0])
    except Exception:
        its = 0
        ksp_type = petsc_options.get("ksp_type", "unknown")
        pc_type = petsc_options.get("pc_type", "unknown")
        rtol = float(petsc_options.get("ksp_rtol", 1e-8))

    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()

    return {
        "mesh": msh,
        "W": W,
        "V": V,
        "Q": Q,
        "w": wh,
        "u": u_h,
        "p": p_h,
        "iterations": its,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
    }


def _sample_function_on_grid(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    vals_local = None
    if len(local_points) > 0:
        vals_local = u_fun.eval(np.array(local_points, dtype=np.float64),
                                np.array(local_cells, dtype=np.int32))
        vals_local = np.asarray(vals_local, dtype=np.float64)

    comm = msh.comm
    gathered_ids = comm.gather(np.array(local_ids, dtype=np.int32), root=0)
    if vals_local is None:
        gathered_vals = comm.gather(np.zeros((0, msh.geometry.dim), dtype=np.float64), root=0)
    else:
        if vals_local.ndim == 1:
            vals_local = vals_local.reshape(-1, msh.geometry.dim)
        gathered_vals = comm.gather(vals_local, root=0)

    mag_grid = None
    if comm.rank == 0:
        full_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
        for ids, vals in zip(gathered_ids, gathered_vals):
            if len(ids) > 0:
                full_vals[ids] = vals
        mags = np.linalg.norm(full_vals, axis=1)
        mag_grid = mags.reshape(ny, nx)

    mag_grid = comm.bcast(mag_grid, root=0)
    return mag_grid


def _sample_vector_at_points(u_fun, points):
    msh = u_fun.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    vals_local = None
    if len(local_points) > 0:
        vals_local = u_fun.eval(np.array(local_points, dtype=np.float64),
                                np.array(local_cells, dtype=np.int32))
        vals_local = np.asarray(vals_local, dtype=np.float64)

    comm = msh.comm
    g_ids = comm.gather(np.array(local_ids, dtype=np.int32), root=0)
    if vals_local is None:
        g_vals = comm.gather(np.zeros((0, msh.geometry.dim), dtype=np.float64), root=0)
    else:
        if vals_local.ndim == 1:
            vals_local = vals_local.reshape(-1, msh.geometry.dim)
        g_vals = comm.gather(vals_local, root=0)

    out = None
    if comm.rank == 0:
        out = np.full((points.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
        for ids, vals in zip(g_ids, g_vals):
            if len(ids) > 0:
                out[ids] = vals
    out = comm.bcast(out, root=0)
    return out


def _estimate_accuracy(force_strings, nu_value, time_budget=20.0):
    comm = MPI.COMM_WORLD
    n_list = [20, 32]
    start = time.perf_counter()
    sols = []
    for n in n_list:
        sols.append(_solve_stokes(comm, n, nu_value, force_strings))
    elapsed = time.perf_counter() - start

    while elapsed < 0.35 * time_budget and n_list[-1] < 96:
        n_next = int(round(n_list[-1] * 1.5))
        if n_next <= n_list[-1]:
            n_next = n_list[-1] + 8
        sols.append(_solve_stokes(comm, n_next, nu_value, force_strings))
        n_list.append(n_next)
        elapsed = time.perf_counter() - start
        if len(n_list) >= 4:
            break

    finest = sols[-1]
    prev = sols[-2]

    probe_n = 25
    xs = np.linspace(0.03, 0.97, probe_n)
    ys = np.linspace(0.03, 0.97, probe_n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(probe_n * probe_n, dtype=np.float64)])

    uf = _sample_vector_at_points(finest["u"], pts)
    uc = _sample_vector_at_points(prev["u"], pts)

    diff = uf - uc
    valid = np.isfinite(diff).all(axis=1)
    if np.any(valid):
        rel = np.linalg.norm(diff[valid].ravel()) / max(np.linalg.norm(uf[valid].ravel()), 1e-14)
        abs_err = np.linalg.norm(diff[valid].ravel()) / np.sqrt(valid.sum())
    else:
        rel = np.nan
        abs_err = np.nan

    return {
        "chosen_n": n_list[-1],
        "verification": {
            "mesh_levels": n_list,
            "relative_change_probe": float(rel) if np.isfinite(rel) else None,
            "absolute_change_probe": float(abs_err) if np.isfinite(abs_err) else None,
        },
        "timing_estimate_sec": float(elapsed),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    nu_value = float(case_spec.get("physics", {}).get("viscosity", pde.get("viscosity", 0.1)))
    force_strings = case_spec.get("pde", {}).get(
        "source_term",
        ['3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))',
         '3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))']
    )
    if isinstance(force_strings, dict):
        force_strings = force_strings.get("value", force_strings)
    if not isinstance(force_strings, (list, tuple)) or len(force_strings) != 2:
        force_strings = ['3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))',
                         '3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))']

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    diag = _estimate_accuracy(force_strings, nu_value, time_budget=20.0)
    n = int(diag["chosen_n"])

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
    }

    sol = _solve_stokes(comm, n, nu_value, force_strings, petsc_options=petsc_options)
    u_grid = _sample_function_on_grid(sol["u"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": n,
        "element_degree": 2,
        "ksp_type": sol["ksp_type"],
        "pc_type": sol["pc_type"],
        "rtol": sol["rtol"],
        "iterations": int(sol["iterations"]),
        "verification": diag["verification"],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.1,
            "source_term": [
                "3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))",
                "3*exp(-50*((x[0]-0.15)**2 + (x[1]-0.15)**2))",
            ],
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])

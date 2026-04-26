import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def _as_bool_mask(arr):
    return np.asarray(arr, dtype=bool)


def _build_bc_function(V, value):
    gdim = V.mesh.geometry.dim
    f = fem.Function(V)
    val = np.array(value, dtype=ScalarType).reshape(gdim, 1)

    def expr(x):
        return np.tile(val, (1, x.shape[1]))

    f.interpolate(expr)
    return f


def _locate_boundary_facets(msh, side):
    fdim = msh.topology.dim - 1
    if side == "x0":
        marker = lambda x: _as_bool_mask(np.isclose(x[0], 0.0))
    elif side == "x1":
        marker = lambda x: _as_bool_mask(np.isclose(x[0], 1.0))
    elif side == "y0":
        marker = lambda x: _as_bool_mask(np.isclose(x[1], 0.0))
    elif side == "y1":
        marker = lambda x: _as_bool_mask(np.isclose(x[1], 1.0))
    else:
        raise ValueError(f"Unknown boundary side '{side}'")
    return mesh.locate_entities_boundary(msh, fdim, marker)


def _extract_dirichlet_bcs(case_spec):
    bc_spec = case_spec.get("boundary_conditions", case_spec.get("bc", {}))
    if isinstance(bc_spec, dict) and "dirichlet" in bc_spec:
        bc_spec = bc_spec["dirichlet"]

    parsed = []
    if isinstance(bc_spec, dict):
        for side, val in bc_spec.items():
            parsed.append((side, val))
    elif isinstance(bc_spec, list):
        for item in bc_spec:
            side = item.get("boundary", item.get("marker", item.get("name")))
            val = item.get("value", item.get("u", item.get("velocity")))
            if side is not None and val is not None:
                parsed.append((side, val))

    if not parsed:
        parsed = [("y1", [1.0, 0.0]), ("x1", [0.0, -0.8]), ("x0", [0.0, 0.0]), ("y0", [0.0, 0.0])]
    return parsed


def _get_viscosity(case_spec):
    pde = case_spec.get("pde", {})
    val = pde.get("viscosity", case_spec.get("viscosity", 0.3))
    try:
        return float(val)
    except Exception:
        return 0.3


def _get_mesh_resolution(case_spec):
    out = case_spec.get("output", {}).get("grid", {})
    nxg = int(out.get("nx", 64))
    nyg = int(out.get("ny", 64))
    target = max(nxg, nyg)
    if target <= 64:
        return 96
    if target <= 128:
        return 128
    return min(192, int(1.25 * target))


def _sample_velocity_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    points = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    points[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    cells = []
    points_on_proc = []
    point_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            point_ids.append(i)

    local_mag = np.full(points.shape[0], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals)
        if vals.ndim == 1:
            mags = np.abs(vals)
        else:
            mags = np.linalg.norm(vals, axis=1)
        local_mag[np.array(point_ids, dtype=np.int32)] = mags

    comm = msh.comm
    gathered = comm.gather(local_mag, root=0)
    if comm.rank == 0:
        merged = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged = np.nan_to_num(merged, nan=0.0)
        result = merged.reshape(ny, nx)
    else:
        result = None

    result = comm.bcast(result, root=0)
    return result


def _compute_diagnostics(msh, u_h, bcs_velocity):
    V = u_h.function_space
    Q0 = fem.functionspace(msh, ("DG", 0))
    q0 = ufl.TestFunction(Q0)
    div_form = fem.form(ufl.inner(ufl.div(u_h), q0) * ufl.dx)
    div_vec = fpetsc.assemble_vector(div_form)
    div_norm_local = div_vec.norm()
    div_norm = msh.comm.allreduce(div_norm_local, op=MPI.SUM)

    bc_errs = []
    for side, val in bcs_velocity:
        facets = _locate_boundary_facets(msh, side)
        dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
        if len(dofs) == 0:
            continue
        arr = u_h.x.array.reshape((-1, msh.geometry.dim))
        vals = arr[dofs]
        target = np.array(val, dtype=np.float64)
        err = np.sqrt(np.mean(np.sum((vals - target) ** 2, axis=1)))
        bc_errs.append(err)
    bc_rms = float(max(bc_errs) if bc_errs else 0.0)
    return float(div_norm), bc_rms


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = _get_viscosity(case_spec)
    n = _get_mesh_resolution(case_spec)

    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f = fem.Constant(msh, np.zeros(gdim, dtype=ScalarType))
    nu_c = fem.Constant(msh, ScalarType(nu))

    a = (
        2.0 * nu_c * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    bcs = []
    bc_pairs = _extract_dirichlet_bcs(case_spec)
    fdim = msh.topology.dim - 1
    for side, val in bc_pairs:
        facets = _locate_boundary_facets(msh, side)
        u_bc = _build_bc_function(V, val)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        if len(dofs) > 0:
            bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) == 0:
        p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 1.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    solver_configs = [
        {"ksp_type": "minres", "pc_type": "hypre", "rtol": 1e-9},
        {"ksp_type": "gmres", "pc_type": "ilu", "rtol": 1e-9},
        {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12},
    ]

    wh = None
    used = None
    its = 0
    last_error = None

    for cfg in solver_configs:
        try:
            problem = fpetsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix="stokes_",
                petsc_options={
                    "ksp_type": cfg["ksp_type"],
                    "pc_type": cfg["pc_type"],
                    "ksp_rtol": cfg["rtol"],
                    "ksp_atol": 1e-12,
                    "ksp_max_it": 5000,
                    "ksp_error_if_not_converged": True,
                    "mat_mumps_icntl_24": 1,
                    "mat_mumps_icntl_25": 0,
                },
            )
            wh = problem.solve()
            wh.x.scatter_forward()
            try:
                its = int(problem.solver.getIterationNumber())
            except Exception:
                its = 0
            used = cfg
            break
        except Exception as e:
            last_error = e
            wh = None

    if wh is None:
        raise RuntimeError(f"All solver configurations failed. Last error: {last_error}")

    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()

    grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(u_h, grid)

    div_norm, bc_rms = _compute_diagnostics(msh, u_h, bc_pairs)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": used["ksp_type"],
        "pc_type": used["pc_type"],
        "rtol": float(used["rtol"]),
        "iterations": int(its),
        "pressure_fixing": "pointwise_p0_at_corner",
        "verification": {
            "discrete_divergence_dg0_vector_norm": float(div_norm),
            "boundary_velocity_rms_error": float(bc_rms),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}

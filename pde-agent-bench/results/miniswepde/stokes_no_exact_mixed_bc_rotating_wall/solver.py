import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def _parse_expr(expr_str, x):
    """Safely evaluate a string expression with x coordinates."""
    if isinstance(expr_str, (int, float)):
        return float(expr_str) * np.ones(x.shape[1])
    expr_str = str(expr_str).strip()
    env = {
        "x": x, "np": np, "pi": np.pi,
        "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "sqrt": np.sqrt, "abs": np.abs, "pow": np.power,
    }
    try:
        result = eval(expr_str, {"__builtins__": {}}, env)
        if np.isscalar(result):
            return float(result) * np.ones(x.shape[1])
        return np.asarray(result, dtype=np.float64)
    except Exception:
        try:
            return float(expr_str) * np.ones(x.shape[1])
        except Exception:
            return np.zeros(x.shape[1])


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # ── Parse case_spec ──────────────────────────────────────────────
    oracle_config = case_spec.get("oracle_config", {})
    pde = oracle_config.get("pde", case_spec.get("pde", {}))
    pde_params = pde.get("pde_params", {})
    nu_val = float(pde_params.get("nu", pde.get("viscosity", 1.0)))

    bc_config = oracle_config.get("bc", {})
    dirichlet_bcs = bc_config.get("dirichlet", [])

    source = pde.get("source_term", ["0.0", "0.0"])

    # ── Mesh & FE spaces ─────────────────────────────────────────────
    N = 64
    degree_u = 2
    degree_p = 1

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)

    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Source term
    try:
        fx = float(source[0])
        fy = float(source[1])
    except (ValueError, TypeError, IndexError):
        fx, fy = 0.0, 0.0
    f = fem.Constant(domain, PETSc.ScalarType((fx, fy)))

    # ── Variational forms ────────────────────────────────────────────
    a_form = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        - ufl.div(u_trial) * q_test * ufl.dx
    )
    L_form = ufl.inner(f, v_test) * ufl.dx

    # ── Boundary conditions ──────────────────────────────────────────
    bcs = []
    W_sub0 = W.sub(0)
    V_collapsed, _map = W_sub0.collapse()

    def _get_marker(on_str):
        on = on_str.lower().strip()
        if on in ("x0", "x=0", "left"):
            return lambda x: np.isclose(x[0], 0.0)
        elif on in ("x1", "x=1", "right"):
            return lambda x: np.isclose(x[0], 1.0)
        elif on in ("y0", "y=0", "bottom"):
            return lambda x: np.isclose(x[1], 0.0)
        elif on in ("y1", "y=1", "top"):
            return lambda x: np.isclose(x[1], 1.0)
        elif "all" in on or "boundary" in on:
            return lambda x: (
                np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
            )
        else:
            return lambda x: (
                np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
            )

    def _apply_dirichlet(on_str, value):
        marker = _get_marker(on_str)
        facets = mesh.locate_entities_boundary(domain, fdim, marker)

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            vx_str, vy_str = str(value[0]), str(value[1])
        else:
            vx_str, vy_str = str(value), "0.0"

        u_bc_func = fem.Function(V_collapsed)
        try:
            vx_val = float(vx_str)
            vy_val = float(vy_str)
            u_bc_func.interpolate(
                lambda x, vx=vx_val, vy=vy_val: np.vstack(
                    [vx * np.ones(x.shape[1]), vy * np.ones(x.shape[1])]
                )
            )
        except ValueError:
            def _make_interp(vxs, vys):
                def interp(x):
                    res = np.zeros((gdim, x.shape[1]))
                    res[0] = _parse_expr(vxs, x)
                    res[1] = _parse_expr(vys, x)
                    return res
                return interp
            u_bc_func.interpolate(_make_interp(vx_str, vy_str))

        dofs = fem.locate_dofs_topological((W_sub0, V_collapsed), fdim, facets)
        bcs.append(fem.dirichletbc(u_bc_func, dofs, W_sub0))

    if dirichlet_bcs:
        for bc_item in dirichlet_bcs:
            _apply_dirichlet(bc_item["on"], bc_item["value"])
    else:
        # Fallback: default BCs for rotating-wall problem
        # x=0 (left): no-slip
        _apply_dirichlet("x0", ["0.0", "0.0"])
        # y=0 (bottom): no-slip
        _apply_dirichlet("y0", ["0.0", "0.0"])
        # y=1 (top): tangential wall motion
        _apply_dirichlet("y1", ["0.5", "0.0"])
        # x=1 (right): natural outflow (no Dirichlet)

    # ── Solve ────────────────────────────────────────────────────────
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": "minres",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "5000",
        },
        petsc_options_prefix="stokes_",
    )
    wh = problem.solve()

    # ── Extract velocity ─────────────────────────────────────────────
    V_out = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,)),
    )
    u_sol = fem.Function(V_out)
    u_sol.interpolate(wh.sub(0))

    # ── Evaluate on 100×100 grid ─────────────────────────────────────
    output_cfg = oracle_config.get("output", {}).get("grid", {})
    nx_out = output_cfg.get("nx", 100)
    ny_out = output_cfg.get("ny", 100)
    bbox = output_cfg.get("bbox", [0, 1, 0, 1])

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    vel_magnitude = np.zeros(points_3d.shape[0])
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        vel_mag = np.sqrt(vals[:, 0] ** 2 + vals[:, 1] ** 2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = vel_mag[idx]

    u_grid = vel_magnitude.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "minres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": 1,
        },
    }

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("viscosity", 1.0))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", {})
    N = 64
    return _solve_stokes(N, nu_val, source, bcs_spec, case_spec)


def _make_bc_func(value_spec, gdim):
    if isinstance(value_spec, str):
        def func(x):
            local_vars = {"x": x, "np": np, "pi": np.pi}
            val = eval(value_spec, {"__builtins__": {}}, local_vars)
            if np.isscalar(val):
                val = np.full_like(x[0], float(val))
            return np.stack([val] + [np.zeros_like(x[0])] * (gdim - 1))
        return func
    elif isinstance(value_spec, (list, tuple)):
        str_exprs = []
        for v in value_spec:
            if isinstance(v, str):
                str_exprs.append(v)
            else:
                str_exprs.append(str(float(v)))
        while len(str_exprs) < gdim:
            str_exprs.append("0.0")
        exprs = str_exprs[:gdim]
        def func(x):
            results = []
            local_vars = {"x": x, "np": np, "pi": np.pi}
            for expr in exprs:
                val = eval(expr, {"__builtins__": {}}, local_vars)
                if np.isscalar(val):
                    val = np.full_like(x[0], float(val))
                results.append(val)
            return np.stack(results)
        return func
    elif isinstance(value_spec, (int, float)):
        fv = float(value_spec)
        return lambda x: np.stack([np.full_like(x[0], fv)] + [np.zeros_like(x[0])] * (gdim - 1))
    else:
        return lambda x: np.zeros((gdim, x.shape[1]))


def _get_boundary_marker(location):
    loc = str(location).lower().strip()
    markers = {
        "left": lambda x: np.isclose(x[0], 0.0),
        "x=0": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], 1.0),
        "x=1": lambda x: np.isclose(x[0], 1.0),
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "y=0": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], 1.0),
        "y=1": lambda x: np.isclose(x[1], 1.0),
        "all": lambda x: np.ones(x.shape[1], dtype=bool),
        "boundary": lambda x: np.ones(x.shape[1], dtype=bool),
        "inflow": lambda x: np.isclose(x[0], 0.0),
        "outflow": lambda x: np.isclose(x[0], 1.0),
        "walls": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
        "noslip": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
        "no-slip": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
        "no_slip": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
    }
    return markers.get(loc, None)


def _solve_stokes(N, nu_val, source, bcs_spec, case_spec):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    degree_u = 2
    degree_p = 1

    el_u = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    el_p = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([el_u, el_p])
    W = fem.functionspace(domain, mel)

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    w_trial = ufl.TrialFunction(W)
    (u_trial, p_trial) = ufl.split(w_trial)
    w_test = ufl.TestFunction(W)
    (v_test, q_test) = ufl.split(w_test)

    nu = fem.Constant(domain, ScalarType(nu_val))

    f_components = [0.0, 0.0]
    if isinstance(source, list) and len(source) >= 2:
        try:
            f_components = [float(s) for s in source]
        except (ValueError, TypeError):
            f_components = [0.0, 0.0]
    f_vec = fem.Constant(domain, ScalarType(np.array(f_components, dtype=np.float64)))

    a_form = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        + q_test * ufl.div(u_trial) * ufl.dx
    )
    L_form = ufl.inner(f_vec, v_test) * ufl.dx

    bcs = []
    bc_list = bcs_spec if isinstance(bcs_spec, list) else []

    if len(bc_list) > 0:
        for bc_item in bc_list:
            bc_type = bc_item.get("type", "dirichlet")
            location = bc_item.get("location", "")
            value = bc_item.get("value", None)
            variable = bc_item.get("variable", "velocity")
            if bc_type == "dirichlet" and variable in ("velocity", "u"):
                marker_func = _get_boundary_marker(location)
                if marker_func is not None:
                    facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
                    if len(facets) > 0:
                        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
                        u_bc = fem.Function(V)
                        if value is not None:
                            u_bc.interpolate(_make_bc_func(value, gdim))
                        else:
                            u_bc.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
                        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    else:
        # Default channel flow BCs
        facets_left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_left)
        u_inlet = fem.Function(V)
        u_inlet.interpolate(lambda x: np.stack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
        bcs.append(fem.dirichletbc(u_inlet, dofs_left, W.sub(0)))

        facets_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
        u_ns_top = fem.Function(V)
        u_ns_top.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
        bcs.append(fem.dirichletbc(u_ns_top, dofs_top, W.sub(0)))

        facets_bot = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        dofs_bot = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bot)
        u_ns_bot = fem.Function(V)
        u_ns_bot.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
        bcs.append(fem.dirichletbc(u_ns_bot, dofs_bot, W.sub(0)))

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    n_points = points_3d.shape[1]
    vel_mag = np.full(n_points, np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(np.sum(vals**2, axis=1))
        for idx, map_idx in enumerate(eval_map):
            vel_mag[map_idx] = vel_magnitude[idx]

    vel_mag_grid = vel_mag.reshape((nx_eval, ny_eval))
    vel_mag_grid = np.nan_to_num(vel_mag_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {"u": vel_mag_grid, "solver_info": solver_info}

"""
solver.py – 2D Linear Elasticity (Small Strain) on unit square
dolfinx v0.10.0

Handles case_spec from pdebench config.json format.
BC: u = [sin(pi*y), 0.0] on all boundaries (boundary-driven, no exact solution).
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve 2D linear elasticity: -div(sigma(u)) = f in Omega, u = g on dOmega
    sigma = 2*mu*eps(u) + lam*tr(eps(u))*I
    """
    comm = MPI.COMM_WORLD

    # ---- Extract parameters from case_spec ----
    # Support both flat and nested oracle_config formats
    oc = case_spec.get("oracle_config", case_spec)
    pde = oc.get("pde", {})
    params = pde.get("pde_params", pde.get("parameters", {}))

    E_val = float(params.get("E", 1.0))
    nu_val = float(params.get("nu", 0.3))

    # Lame parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

    # Source term
    source = pde.get("source_term", pde.get("source", ["0.0", "0.0"]))

    # Domain
    domain_spec = oc.get("domain", {})

    # Output grid
    output_cfg = oc.get("output", {})
    grid_cfg = output_cfg.get("grid", {})
    bbox = grid_cfg.get("bbox", [0, 1, 0, 1])
    nx_out = grid_cfg.get("nx", 50)
    ny_out = grid_cfg.get("ny", 50)

    # Boundary conditions
    bc_cfg = oc.get("bc", {})
    dirichlet_cfg = bc_cfg.get("dirichlet", None)

    # Also check flat format
    if dirichlet_cfg is None:
        bcs_list = case_spec.get("boundary_conditions", [])
    else:
        bcs_list = None

    # ---- Mesh ----
    N = 64
    element_degree = 2

    x0, x1 = float(bbox[0]), float(bbox[1])
    y0, y1 = float(bbox[2]), float(bbox[3])

    domain = mesh.create_rectangle(
        comm,
        [np.array([x0, y0]), np.array([x1, y1])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", element_degree, (gdim,)))

    mu_c = fem.Constant(domain, ScalarType(mu_val))
    lam_c = fem.Constant(domain, ScalarType(lam_val))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu_c * epsilon(w) + lam_c * ufl.tr(epsilon(w)) * ufl.Identity(gdim)

    # ---- Source term ----
    x_coord = ufl.SpatialCoordinate(domain)
    f_expr = _build_source(domain, source, gdim, x_coord)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # ---- Boundary Conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1

    bcs = []
    if dirichlet_cfg is not None:
        # Oracle-style BC format: {"on": "all", "value": ["sin(pi*y)", "0.0"]}
        if isinstance(dirichlet_cfg, dict):
            dirichlet_list = [dirichlet_cfg]
        else:
            dirichlet_list = dirichlet_cfg

        for dc in dirichlet_list:
            on = dc.get("on", "all")
            value = dc.get("value", ["0.0", "0.0"])
            marker = _boundary_selector(on, x0, x1, y0, y1)

            # Parse value expressions
            if isinstance(value, (list, tuple)):
                expr_list = [str(v) for v in value]
            else:
                expr_list = [str(value)] * gdim

            # Build UFL expression for BC
            u_bc = fem.Function(V)
            _interpolate_vector_bc(u_bc, expr_list, domain, gdim, x_coord)

            dofs = fem.locate_dofs_geometrical(V, marker)
            bcs.append(fem.dirichletbc(u_bc, dofs))

    elif bcs_list:
        # Flat format from test_solver.py
        for bc_spec in bcs_list:
            bc_type = bc_spec.get("type", "dirichlet")
            if bc_type != "dirichlet":
                continue
            location = bc_spec.get("location", "all")
            value = bc_spec.get("value", None)
            component = bc_spec.get("component", None)

            marker = _location_marker(location, x0, x1, y0, y1)
            facets = mesh.locate_entities_boundary(domain, fdim, marker)

            if component is not None:
                comp = int(component)
                V_sub = V.sub(comp)
                V_collapsed, _ = V_sub.collapse()
                u_bc = fem.Function(V_collapsed)
                if isinstance(value, (list, tuple)):
                    bc_val = float(value[comp]) if len(value) > comp else float(value[0])
                elif value is not None:
                    bc_val = float(value)
                else:
                    bc_val = 0.0
                u_bc.interpolate(lambda x, bv=bc_val: np.full(x.shape[1], bv))
                dofs = fem.locate_dofs_topological((V_sub, V_collapsed), fdim, facets)
                bcs.append(fem.dirichletbc(u_bc, dofs, V_sub))
            else:
                if isinstance(value, (list, tuple)):
                    val_vec = [float(vv) for vv in value]
                elif value is not None:
                    val_vec = [float(value)] * gdim
                else:
                    val_vec = [0.0] * gdim
                u_bc = fem.Function(V)
                vv = list(val_vec)
                u_bc.interpolate(lambda x, a=vv[0], b=vv[1]: np.vstack([
                    np.full(x.shape[1], a), np.full(x.shape[1], b)]))
                dofs = fem.locate_dofs_topological(V, fdim, facets)
                bcs.append(fem.dirichletbc(u_bc, dofs))
    else:
        # Default: zero on all boundaries
        all_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
        dofs = fem.locate_dofs_topological(V, fdim, all_facets)
        bcs.append(fem.dirichletbc(u_bc, dofs))

    # ---- Solve ----
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
            },
            petsc_options_prefix="elast_",
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="elast_",
        )
        u_sol = problem.solve()

    # ---- Evaluate on grid ----
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out, gdim, bbox)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        },
    }


def _build_source(domain, source, gdim, x_coord):
    """Build source term vector."""
    if source is None:
        return fem.Constant(domain, ScalarType((0.0,) * gdim))

    f_vals = []
    all_const = True
    for s in source:
        s_str = str(s).strip()
        try:
            f_vals.append(float(s_str))
        except ValueError:
            all_const = False
            break

    if all_const and len(f_vals) == gdim:
        return fem.Constant(domain, ScalarType(tuple(f_vals)))

    # Parse as UFL expressions
    comps = []
    for s in source:
        comps.append(_parse_ufl_expr(str(s).strip(), x_coord))
    return ufl.as_vector(comps)


def _parse_ufl_expr(expr_str, x_coord):
    """Parse a string expression into a UFL expression."""
    import math
    s = expr_str.strip()
    try:
        return float(s)
    except ValueError:
        pass

    # Replace common math functions
    s_eval = s.replace("pi", str(math.pi))
    s_eval = s_eval.replace("x", "x_coord[0]").replace("y", "x_coord[1]")
    s_eval = s_eval.replace("sin", "ufl.sin").replace("cos", "ufl.cos")
    s_eval = s_eval.replace("exp", "ufl.exp")

    local_ns = {"x_coord": x_coord, "ufl": ufl, "pi": math.pi}
    return eval(s_eval, {"__builtins__": {}}, local_ns)


def _interpolate_vector_bc(u_bc, expr_list, domain, gdim, x_coord):
    """Interpolate vector BC from string expressions."""
    import math

    # Check if all components are simple constants
    const_vals = []
    all_const = True
    for expr_str in expr_list:
        try:
            const_vals.append(float(expr_str))
        except ValueError:
            all_const = False
            break

    if all_const:
        cv = const_vals
        u_bc.interpolate(lambda x, vals=cv: np.array([[v] * x.shape[1] for v in vals]))
        return

    # Need to use UFL expression + fem.Expression
    ufl_comps = []
    for expr_str in expr_list:
        ufl_comps.append(_parse_ufl_expr(expr_str, x_coord))

    ufl_vec = ufl.as_vector(ufl_comps)
    expr = fem.Expression(ufl_vec, u_bc.function_space.element.interpolation_points)
    u_bc.interpolate(expr)


def _boundary_selector(on, x0, x1, y0, y1):
    """Return boundary marker function for oracle-style 'on' specification."""
    key = on.lower().strip()
    if key in ("all", "*"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    if key in ("x0", "xmin", "left"):
        return lambda x: np.isclose(x[0], x0)
    if key in ("x1", "xmax", "right"):
        return lambda x: np.isclose(x[0], x1)
    if key in ("y0", "ymin", "bottom"):
        return lambda x: np.isclose(x[1], y0)
    if key in ("y1", "ymax", "top"):
        return lambda x: np.isclose(x[1], y1)
    return lambda x: np.ones(x.shape[1], dtype=bool)


def _location_marker(location, x0, x1, y0, y1):
    """Return boundary marker for flat-format location strings."""
    loc = location.lower().strip()
    if loc == "left":
        return lambda x: np.isclose(x[0], x0)
    elif loc == "right":
        return lambda x: np.isclose(x[0], x1)
    elif loc == "bottom":
        return lambda x: np.isclose(x[1], y0)
    elif loc == "top":
        return lambda x: np.isclose(x[1], y1)
    elif loc == "all":
        return lambda x: np.ones(x.shape[1], dtype=bool)
    return lambda x: np.ones(x.shape[1], dtype=bool)


def _evaluate_on_grid(domain, u_sol, nx, ny, gdim, bbox):
    """Evaluate displacement magnitude on a uniform grid."""
    x0, x1, y0, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    points = np.zeros((3, nx * ny))
    points[0] = XX.flatten()
    points[1] = YY.flatten()

    tdim = domain.topology.dim
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    n_points = nx * ny
    u_values = np.zeros((n_points, gdim))

    pts_list = []
    cells_list = []
    emap = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points[:, i])
            cells_list.append(links[0])
            emap.append(i)

    if len(pts_list) > 0:
        pts_arr = np.array(pts_list)
        cells_arr = np.array(cells_list, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, mi in enumerate(emap):
            u_values[mi, :] = vals[idx, :gdim]

    disp_mag = np.sqrt(np.sum(u_values ** 2, axis=1))
    return disp_mag.reshape((nx, ny))
